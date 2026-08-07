// Local validation driver for Learning-CUDA kernels (Windows host).
// Compares kernels.cu implementations against torch-generated references.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <cuda_fp16.h>

// Include the student implementation (with template instantiations).
#include "../src/kernels.cu"

// ------------------------------------------------------------------
static std::vector<unsigned char> read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> buf(sz);
    fread(buf.data(), 1, sz, f);
    fclose(f);
    return buf;
}

template <typename T>
static double max_diff(const std::vector<T>& a, const std::vector<T>& b) {
    double md = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        float va = (float)a[i], vb = (float)b[i];
        md = std::max(md, (double)std::fabs(va - vb));
    }
    return md;
}

template <typename T>
static double max_rel(const std::vector<T>& a, const std::vector<T>& b) {
    double md = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        float va = (float)a[i], vb = (float)b[i];
        double d = std::fabs(va - vb);
        if (vb != 0.0f) d = std::fabs((va - vb) / vb);
        md = std::max(md, d);
    }
    return md;
}

// ------------------------------------------------------------------
static int test_rms(const std::string& dir, int idx, bool f16) {
    std::string suf = f16 ? "_f16_" : "_f32_";
    auto in_b = read_file(dir + "/rms_input" + suf + std::to_string(idx) + ".bin");
    auto w_b  = read_file(dir + "/rms_weight" + suf + std::to_string(idx) + ".bin");
    auto ref_b= read_file(dir + "/rms_ref" + suf + std::to_string(idx) + ".bin");
    FILE* mf = fopen((dir + "/rms_meta_" + std::to_string(idx) + ".txt").c_str(), "r");
    size_t rows, hidden; float eps;
    fscanf(mf, "%zu %zu %f", &rows, &hidden, &eps);
    fclose(mf);

    int ok = 1;
    if (f16) {
        std::vector<half> in((half*)in_b.data(), (half*)in_b.data() + in_b.size()/2);
        std::vector<half> w((half*)w_b.data(), (half*)w_b.data() + w_b.size()/2);
        std::vector<half> out(rows * hidden);
        std::vector<half> ref((half*)ref_b.data(), (half*)ref_b.data() + ref_b.size()/2);
        rmsNorm(in, w, out, rows, hidden, eps);
        double md = max_diff(out, ref);
        printf("  rmsNorm f16 case%d rows=%zu hidden=%zu max_abs=%.6f %s\n",
               idx, rows, hidden, md, md < 0.05 ? "PASS" : "FAIL");
        if (md >= 0.05) ok = 0;
    } else {
        std::vector<float> in((float*)in_b.data(), (float*)in_b.data() + in_b.size()/4);
        std::vector<float> w((float*)w_b.data(), (float*)w_b.data() + w_b.size()/4);
        std::vector<float> out(rows * hidden);
        std::vector<float> ref((float*)ref_b.data(), (float*)ref_b.data() + ref_b.size()/4);
        rmsNorm(in, w, out, rows, hidden, eps);
        double md = max_diff(out, ref);
        printf("  rmsNorm f32 case%d rows=%zu hidden=%zu max_abs=%.6f %s\n",
               idx, rows, hidden, md, md < 1e-4 ? "PASS" : "FAIL");
        if (md >= 1e-4) ok = 0;
    }
    return ok;
}

// ------------------------------------------------------------------
static int test_attn(const std::string& dir, int idx, bool f16) {
    std::string suf = f16 ? "_f16_" : "_f32_";
    auto q_b = read_file(dir + "/attn_q" + suf + std::to_string(idx) + ".bin");
    auto k_b = read_file(dir + "/attn_k" + suf + std::to_string(idx) + ".bin");
    auto v_b = read_file(dir + "/attn_v" + suf + std::to_string(idx) + ".bin");
    auto ref_b= read_file(dir + "/attn_ref" + suf + std::to_string(idx) + ".bin");
    FILE* mf = fopen((dir + "/attn_meta_" + std::to_string(idx) + ".txt").c_str(), "r");
    int b, tgt, src, qh, kvh, hd, causal;
    fscanf(mf, "%d %d %d %d %d %d %d", &b, &tgt, &src, &qh, &kvh, &hd, &causal);
    fclose(mf);

    int ok = 1;
    if (f16) {
        std::vector<half> q((half*)q_b.data(), (half*)q_b.data() + q_b.size()/2);
        std::vector<half> k((half*)k_b.data(), (half*)k_b.data() + k_b.size()/2);
        std::vector<half> v((half*)v_b.data(), (half*)v_b.data() + v_b.size()/2);
        std::vector<half> o(q.size());
        std::vector<half> ref((half*)ref_b.data(), (half*)ref_b.data() + ref_b.size()/2);
        flashAttention(q, k, v, o, b, tgt, src, qh, kvh, hd, causal != 0);
        double md = max_diff(o, ref);
        printf("  flashAttn f16 case%d [%d,%d,%d,%d,%d,%d] causal=%d max_abs=%.6f %s\n",
               idx, b, tgt, src, qh, kvh, hd, causal, md, md < 0.08 ? "PASS" : "FAIL");
        if (md >= 0.08) ok = 0;
    } else {
        std::vector<float> q((float*)q_b.data(), (float*)q_b.data() + q_b.size()/4);
        std::vector<float> k((float*)k_b.data(), (float*)k_b.data() + k_b.size()/4);
        std::vector<float> v((float*)v_b.data(), (float*)v_b.data() + v_b.size()/4);
        std::vector<float> o(q.size());
        std::vector<float> ref((float*)ref_b.data(), (float*)ref_b.data() + ref_b.size()/4);
        flashAttention(q, k, v, o, b, tgt, src, qh, kvh, hd, causal != 0);
        double md = max_diff(o, ref);
        printf("  flashAttn f32 case%d [%d,%d,%d,%d,%d,%d] causal=%d max_abs=%.6f %s\n",
               idx, b, tgt, src, qh, kvh, hd, causal, md, md < 1e-4 ? "PASS" : "FAIL");
        if (md >= 1e-4) ok = 0;
    }
    return ok;
}

// ------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "local_test";
    int total_ok = 1;

    printf("=== RMSNorm ===\n");
    for (int i = 0; i < 3; ++i) {
        total_ok &= test_rms(dir, i, false);
        total_ok &= test_rms(dir, i, true);
    }
    printf("=== FlashAttention ===\n");
    for (int i = 0; i < 4; ++i) {
        total_ok &= test_attn(dir, i, false);
        total_ok &= test_attn(dir, i, true);
    }
    printf("\n%s\n", total_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return total_ok ? 0 : 1;
}
