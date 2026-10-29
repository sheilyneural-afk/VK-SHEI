// Terrsvusion OpenCL C 1.2 device kernels: QK / softmax / PV.
// Q: f32 [QH,D]. K/V: raw little-endian BF16/F16/F32 [T,KVH,D].
// Functional GPU reference only; no compression, not FlashAttention.

inline float load_scalar(__global const uchar *data, const uint index, const uint dtype) {
    if (dtype == 3u) {
        // OpenCL device endianness and alignment are checked on the host.
        const uint bits = ((__global const uint *)data)[index];
        return as_float(bits);
    }
    const uint raw = (uint)((__global const ushort *)data)[index];
    if (dtype == 1u) return as_float(raw << 16u); // BF16 exact decode
    const uint sign = (raw & 0x8000u) << 16u;
    const uint e = (raw >> 10u) & 31u;
    const uint fraction = raw & 1023u;
    uint bits;
    if (e == 0u) {
        if (fraction == 0u) bits = sign;
        else {
            uint mantissa = fraction;
            int exponent_unbiased = -14;
            while ((mantissa & 0x400u) == 0u) {mantissa <<= 1u; exponent_unbiased--;}
            bits = sign | ((uint)(exponent_unbiased + 127) << 23u)
                        | ((mantissa & 1023u) << 13u);
        }
    } else if (e == 31u) {
        bits = sign | 0x7f800000u | (fraction << 13u);
    } else {
        bits = sign | ((e - 15u + 127u) << 23u) | (fraction << 13u);
    }
    return as_float(bits); // FP16 -> FP32 exact decode
}

__kernel void terrsvusion_qk(
    __global const float *query, __global const uchar *keys,
    __global float *scores, const uint tokens, const uint query_heads,
    const uint kv_heads, const uint head_dim, const uint key_dtype) {
    const uint i = (uint)get_global_id(0);
    if (i >= tokens * query_heads) return;
    const uint qh = i / tokens;
    const uint t = i % tokens;
    const uint kvh = qh / (query_heads / kv_heads);
    const uint ko = (t * kv_heads + kvh) * head_dim;
    const uint qo = qh * head_dim;
    float dot = 0.0f;
    for (uint d = 0u; d < head_dim; ++d)
        dot += query[qo+d] * load_scalar(keys, ko+d, key_dtype);
    scores[i] = dot * rsqrt((float)head_dim);
}

__kernel void terrsvusion_softmax(__global float *scores,
                                    const uint tokens, const uint query_heads) {
    const uint qh = (uint)get_global_id(0);
    if (qh >= query_heads) return;
    const uint start = qh * tokens;
    float maximum = -INFINITY;
    for (uint t=0u; t<tokens; ++t)
        maximum = fmax(maximum,scores[start+t]);
    float denominator = 0.0f;
    for (uint t=0u; t<tokens; ++t) {
        const float p = exp(scores[start+t]-maximum);
        scores[start+t] = p;
        denominator += p;
    }
    for (uint t=0u; t<tokens; ++t)
        scores[start+t] /= denominator;
}

__kernel void terrsvusion_pv(
    __global const float *probabilities, __global const uchar *values,
    __global float *output, const uint tokens, const uint query_heads,
    const uint kv_heads, const uint head_dim, const uint value_dtype) {
    const uint i = (uint)get_global_id(0);
    if (i >= query_heads*head_dim) return;
    const uint qh=i/head_dim, d=i%head_dim;
    const uint kvh=qh/(query_heads/kv_heads);
    float sum=0.0f;
    for (uint t=0u; t<tokens; ++t)
        sum += probabilities[qh*tokens+t]
            * load_scalar(values,(t*kv_heads+kvh)*head_dim+d,value_dtype);
    output[i]=sum;
}

