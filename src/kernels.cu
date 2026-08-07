#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

// ------------------------------------------------------------------
// Type conversion helpers (device): work for both float and half
// ------------------------------------------------------------------
template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_float<half>(half v) { return __half2float(v); }

template <typename T> __device__ __forceinline__ T to_output(float v);
template <> __device__ __forceinline__ float to_output<float>(float v) { return v; }
template <> __device__ __forceinline__ half to_output<half>(float v) { return __float2half_rn(v); }

// ------------------------------------------------------------------
// rmsNorm kernel: one block per row, block-level reduction
// ------------------------------------------------------------------
template <typename T>
__global__ void rmsNormKernel(const T* __restrict__ input,
                              const T* __restrict__ weight,
                              T* __restrict__ output,
                              size_t hidden_dim, float eps) {
    const size_t row = blockIdx.x;
    const T* in = input + row * hidden_dim;
    T* out = output + row * hidden_dim;

    float sum = 0.0f;
    for (size_t j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
        float v = to_float<T>(in[j]);
        sum += v * v;
    }
    __shared__ float ssum[256];
    ssum[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
        __syncthreads();
    }
    const float mean = ssum[0] / (float)hidden_dim;
    const float rstd = rsqrtf(mean + eps);

    for (size_t j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
        out[j] = to_output<T>(to_float<T>(in[j]) * rstd * to_float<T>(weight[j]));
    }
}

// ------------------------------------------------------------------
// flashAttention: online-softmax, one block per (batch, query-head) pair.
// Supports GQA (query_heads % kv_heads == 0) and causal masking.
// Layouts (row-major, like PyTorch):
//   q: [batch, tgt_len, q_heads, head_dim]
//   k: [batch, src_len, kv_heads, head_dim]
//   v: [batch, src_len, kv_heads, head_dim]
//   o: [batch, tgt_len, q_heads, head_dim]
// ------------------------------------------------------------------
template <typename T>
__global__ void flashAttnKernel(const T* __restrict__ q,
                                const T* __restrict__ k,
                                const T* __restrict__ v,
                                T* __restrict__ o,
                                int target_seq_len, int src_seq_len,
                                int query_heads, int kv_heads, int head_dim,
                                bool is_causal) {
    const int b = blockIdx.y;
    const int qh = blockIdx.x;
    // GQA grouping: consecutive query heads share one kv head
    // (matches torch's repeat_interleave over the head dimension).
    const int kvh = qh / (query_heads / kv_heads);

    // Layouts are PyTorch-style [batch, seq, heads, head_dim].
    const long long q_batch = (long long)b * target_seq_len * query_heads * head_dim;
    const long long kv_batch = (long long)b * src_seq_len * kv_heads * head_dim;
    const T* q_base = q + q_batch;
    const T* k_base = k + kv_batch;
    const T* v_base = v + kv_batch;
    T* o_base = o + q_batch;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    extern __shared__ float sacc[];  // [head_dim] running weighted sum for current query row
    __shared__ float sscore[256];    // reduction buffer for scores

    for (int qi = 0; qi < target_seq_len; ++qi) {
        const long long q_off = (long long)qi * query_heads * head_dim + (long long)qh * head_dim;
        const T* qrow_ptr = q_base + q_off;
        const int max_k = is_causal ? (qi + 1) : src_seq_len;

        // init running accumulator
        for (int d = tid; d < head_dim; d += nthreads) sacc[d] = 0.0f;
        __syncthreads();

        float m = -1e30f;
        float l = 0.0f;

        for (int kj = 0; kj < max_k; ++kj) {
            const long long k_off = (long long)kj * kv_heads * head_dim + (long long)kvh * head_dim;
            float score = 0.0f;
            for (int d = tid; d < head_dim; d += nthreads) {
                score += to_float<T>(qrow_ptr[d]) * to_float<T>(k_base[k_off + d]);
            }
            sscore[tid] = score;
            __syncthreads();
            for (int s = nthreads / 2; s > 0; s >>= 1) {
                if (tid < s) sscore[tid] += sscore[tid + s];
                __syncthreads();
            }
            const float s = sscore[0] * rsqrtf((float)head_dim);

            // online softmax
            const float m_new = fmaxf(m, s);
            const float alpha = __expf(m - m_new);
            const float beta = __expf(s - m_new);
            l = l * alpha + beta;
            m = m_new;

            for (int d = tid; d < head_dim; d += nthreads) {
                sacc[d] = sacc[d] * alpha + beta * to_float<T>(v_base[k_off + d]);
            }
            __syncthreads();
        }

        for (int d = tid; d < head_dim; d += nthreads) {
            o_base[q_off + d] = to_output<T>(sacc[d] / l);
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------
// Host wrappers
// ------------------------------------------------------------------
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
             std::vector<T>& h_output, size_t rows, size_t hidden_dim,
             float eps) {
    T *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr;
    size_t n = rows * hidden_dim;
    RUNTIME_CHECK(cudaMalloc(&d_input, n * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_weight, hidden_dim * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_output, n * sizeof(T)));
    RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_weight, h_weight.data(), hidden_dim * sizeof(T), cudaMemcpyHostToDevice));

    const int threads = 256;
    rmsNormKernel<T><<<rows, threads>>>(d_input, d_weight, d_output, hidden_dim, eps);
    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());

    RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost));
    cudaFree(d_input); cudaFree(d_weight); cudaFree(d_output);
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    size_t q_n = (size_t)batch_size * target_seq_len * query_heads * head_dim;
    size_t kv_n = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
    T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    RUNTIME_CHECK(cudaMalloc(&d_q, q_n * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_k, kv_n * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_v, kv_n * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_o, q_n * sizeof(T)));
    RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_n * sizeof(T), cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_n * sizeof(T), cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_n * sizeof(T), cudaMemcpyHostToDevice));

    dim3 grid(query_heads, batch_size);
    int threads = 128;
    size_t smem = head_dim * sizeof(float);
    flashAttnKernel<T><<<grid, threads, smem>>>(d_q, d_k, d_v, d_o,
        target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal);
    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());

    RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, q_n * sizeof(T), cudaMemcpyDeviceToHost));
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
