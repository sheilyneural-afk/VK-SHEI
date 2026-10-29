// Terrsvusion: full BF16 Llama/MiniCPM forward kernels (OpenCL C 1.2).
// Original model weights preserved as raw BF16. No quantization or Python.
// Inference-only, single autoregressive token at each kernel launch.
// Supports grouped-query attention and Llama SwiGLU, RMSNorm, RoPE.
// The Rust executor validates all layer shapes and tensor formats.

inline float read_bf16(const ushort v) {
    return as_float(((uint)v) << 16u);
}
inline ushort write_bf16(const float v) {
    uint bits = as_uint(v);
    // IEEE round-to-nearest-even bfloat16 (for finite values).
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (ushort)(bits >> 16u);
}

// [vocab, hidden] BF16 embedding table: one work-item per hidden element.
__kernel void llm_embed(
    __global const ushort *embedding,
    __global ushort *out,
    const uint token_id,
    const uint hidden_size
) {
    const uint i = (uint)get_global_id(0);
    if (i < hidden_size)
        out[i] = embedding[token_id * hidden_size + i];
}

// One workgroup per output row. Threads cooperatively reduce a matrix row.
// Local size must be 128. Weight matrix layout is [out_dim, in_dim] BF16.
// The BF16 output matches the logical storage dtype of the original model.
__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_gemv_bf16(
    __global const ushort *weights,
    __global const ushort *x,
    __global ushort *result,
    const uint in_dim,
    const uint out_dim
) {
    const uint row = (uint)get_group_id(0);
    const uint tid = (uint)get_local_id(0);
    if (row >= out_dim) return;
    const uint offset = row * in_dim;
    float sum = 0.0f;
    for (uint column = tid; column < in_dim; column += 128u) {
        sum = fma(
            read_bf16(weights[offset + column]),
            read_bf16(x[column]),
            sum
        );
    }
    __local float partials[128];
    partials[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (tid < stride)
            partials[tid] += partials[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0u)
        result[row] = write_bf16(partials[0]);
}

// Each BF16 RMSNorm is computed from input as FP32 variance, then
// intermediate normalized vector rounds to BF16 before multiplying weights.
__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_rmsnorm(
    __global const ushort *input,
    __global const ushort *weights,
    __global ushort *output,
    const uint dimension,
    const float epsilon
) {
    const uint tid = (uint)get_local_id(0);
    __local float partials[128];
    float s = 0.0f;
    for (uint i = tid; i < dimension; i += 128u) {
        const float value = read_bf16(input[i]);
        s = fma(value, value, s);
    }
    partials[tid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (tid < stride) partials[tid] += partials[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float factor = rsqrt(partials[0] / (float)dimension + epsilon);
    for (uint i = tid; i < dimension; i += 128u) {
        const ushort normalized = write_bf16(read_bf16(input[i]) * factor);
        output[i] = write_bf16(read_bf16(normalized) * read_bf16(weights[i]));
    }
}

// RoPE layout: rotate the first half against second half, as in
// Llama's rotate_half; cos/sin repeated for both halves. One query token.
__kernel void llm_rope(
    __global const ushort *input,
    __global ushort *output,
    const uint num_heads,
    const uint head_dim,
    const uint position,
    const float theta,
    const uint llama3_scaling,
    const float scaling_factor,
    const float low_freq_factor,
    const float high_freq_factor,
    const float original_max_position
) {
    const uint i = (uint)get_global_id(0);
    if (i >= num_heads * head_dim) return;
    const uint head = i / head_dim;
    const uint d = i % head_dim;
    const uint half_dim = head_dim / 2u;
    const uint source = head * head_dim + (d < half_dim ? d + half_dim : d - half_dim);
    const float rotated = d < half_dim
        ? -read_bf16(input[source]) : read_bf16(input[source]);
    const uint frequency_dim = (d % half_dim) * 2u;
    float inv_frequency = exp(-(float)frequency_dim * log(theta) / (float)head_dim);
    if (llama3_scaling != 0u) {
        const float wavelength = 6.28318530717958647692f / inv_frequency;
        const float low_wavelength = original_max_position / low_freq_factor;
        const float high_wavelength = original_max_position / high_freq_factor;
        if (wavelength > low_wavelength) {
            inv_frequency /= scaling_factor;
        } else if (wavelength >= high_wavelength) {
            const float smooth =
                (original_max_position / wavelength - low_freq_factor) /
                (high_freq_factor - low_freq_factor);
            inv_frequency = (1.0f - smooth) * (inv_frequency / scaling_factor)
                + smooth * inv_frequency;
        }
    }
    const float phase = (float)position * inv_frequency;
    const float c = read_bf16(write_bf16(cos(phase)));
    const float s = read_bf16(write_bf16(sin(phase)));
    // BF16 multiply and add stage rounding as the BF16 LibTorch expression.
    const float left = read_bf16(write_bf16(read_bf16(input[i]) * c));
    const float right = read_bf16(write_bf16(rotated * s));
    output[i] = write_bf16(left + right);
}

// BF16 KV [max_tokens,kv_heads,head_dim] buffers live on GPU.
// Q [q_heads,head_dim] of current token. One work item per score.
__kernel void llm_attention_qk(
    __global const ushort *query,
    __global const ushort *key_cache,
    __global float *scores,
    const uint tokens,
    const uint q_heads,
    const uint kv_heads,
    const uint dim
) {
    const uint i = (uint)get_global_id(0);
    if (i >= tokens * q_heads) return;
    const uint qh = i / tokens;
    const uint token = i % tokens;
    const uint kvh = qh / (q_heads / kv_heads);
    const uint ko = (token * kv_heads + kvh) * dim;
    const uint qo = qh * dim;
    float dot = 0.0f;
    for (uint d=0u; d<dim; d++)
        dot = fma(read_bf16(query[qo+d]),read_bf16(key_cache[ko+d]),dot);
    scores[i] = dot * rsqrt((float)dim);
}

__kernel void llm_attention_softmax(
    __global float *scores,
    const uint tokens,
    const uint q_heads
) {
    const uint qh = (uint)get_global_id(0);
    if (qh >= q_heads) return;
    const uint start = qh * tokens;
    float maximum = -INFINITY;
    for (uint t=0u;t<tokens;t++)
        maximum = fmax(maximum,scores[start+t]);
    float denominator = 0.0f;
    for (uint t=0u;t<tokens;t++) {
        float probability = exp(scores[start+t]-maximum);
        scores[start+t] = probability;
        denominator += probability;
    }
    for (uint t=0u;t<tokens;t++) scores[start+t] /= denominator;
}

__kernel void llm_attention_pv(
    __global const float *scores,
    __global const ushort *values,
    __global ushort *output,
    const uint tokens,
    const uint q_heads,
    const uint kv_heads,
    const uint dim
) {
    const uint i = (uint)get_global_id(0);
    if (i >= q_heads * dim) return;
    const uint qh = i / dim;
    const uint d = i % dim;
    const uint kvh = qh / (q_heads / kv_heads);
    float acc = 0.0f;
    for (uint t = 0u; t < tokens; t++) {
        const uint offset = (t * kv_heads + kvh) * dim + d;
        acc = fma(scores[qh*tokens + t],read_bf16(values[offset]),acc);
    }
    output[i] = write_bf16(acc);
}

__kernel void llm_add(
    __global const ushort *a,
    __global const ushort *b,
    __global ushort *output,
    const uint n
) {
    const uint i = (uint)get_global_id(0);
    if (i < n) output[i] = write_bf16(read_bf16(a[i])+read_bf16(b[i]));
}

// Llama SwiGLU: (silu(gate) * up) rounded BF16 after each operation.
__kernel void llm_swiglu(
    __global const ushort *gate,
    __global const ushort *up,
    __global ushort *output,
    const uint n
) {
    const uint i = (uint)get_global_id(0);
    if (i >= n) return;
    const float g = read_bf16(gate[i]);
    const float activated = g / (1.0f + exp(-g));
    const ushort g_bf16 = write_bf16(activated);
    output[i] = write_bf16(read_bf16(g_bf16) * read_bf16(up[i]));
}

// One workgroup reduces all LM-head BF16 logits, preserving first-id ties.
__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_argmax(
    __global const ushort *logits,
    __global uint *result,
    const uint vocab_size
) {
    const uint tid = (uint)get_local_id(0);
    float value = -INFINITY;
    uint index = 0u;
    for (uint i=tid;i<vocab_size;i+=128u) {
        const float l = read_bf16(logits[i]);
        if (l > value || (l == value && i < index)) {
            value = l; index = i;
        }
    }
    __local float values[128];
    __local uint indices[128];
    values[tid] = value; indices[tid] = index;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride=64u;stride>0u;stride >>= 1u) {
        if (tid < stride) {
            const float other = values[tid+stride];
            const uint other_id = indices[tid+stride];
            if (other > values[tid] || (other == values[tid] && other_id < indices[tid])) {
                values[tid] = other; indices[tid] = other_id;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0u) result[0] = indices[0];
}



// GPU-LOD canonical cold-V stage.
// QK and softmax remain the EXACT same kernels used by the dense OpenCL
// path. Values come from the original Rust ColdStore with verified BF16
// token hashes. Each staging chunk is processed in ascending token order.
// This avoids holding historical V on the iGPU; only a bounded stage
// and the accumulator survive across chunks.
__kernel void llm_lod_pv_stage(
    __global const float *probabilities,
    __global const ushort *values_stage,
    __global const uint *selected_token_ids,
    __global float *accumulator,
    const uint stage_count,
    const uint full_tokens,
    const uint q_heads,
    const uint kv_heads,
    const uint dim
) {
    const uint i = (uint)get_global_id(0);
    if (i >= q_heads * dim) return;
    const uint qh = i / dim;
    const uint d = i % dim;
    const uint kvh = qh / (q_heads / kv_heads);
    float acc = accumulator[i];
    for (uint index = 0u; index < stage_count; index++) {
        const uint token = selected_token_ids[index];
        const uint stage_offset = (index * kv_heads + kvh) * dim + d;
        acc = fma(
            probabilities[qh * full_tokens + token],
            read_bf16(values_stage[stage_offset]),
            acc
        );
    }
    accumulator[i] = acc;
}

__kernel void llm_lod_pv_finish(
    __global const float *accumulator,
    __global const float *selected_mass_per_head,
    __global ushort *output,
    const uint q_heads,
    const uint dim
) {
    const uint i = (uint)get_global_id(0);
    if (i >= q_heads * dim) return;
    const uint qh = i / dim;
    const float mass = selected_mass_per_head[qh];
    const float value = accumulator[i] / mass;
    output[i] = write_bf16(value);
}


// ----------------------------------------------------------------------
// Native GPU BATCH PREFILL: SAME BF16 arithmetic and causal semantics as
// one-token kernels above, but process a tile of prompt tokens per layer.
// All matrix rows reuse the original weights on GPU. Original CPU and
// ColdStore code are unchanged. Each batch is strictly causal.
// ----------------------------------------------------------------------
__kernel void llm_prefill_embed(
    __global const ushort *weights,
    __global const uint *token_ids,
    __global ushort *output,
    const uint tokens,
    const uint hidden
) {
    const uint i=(uint)get_global_id(0);
    if(i>=tokens*hidden) return;
    const uint token=i/hidden;
    const uint dim=i%hidden;
    output[i]=weights[token_ids[token]*hidden+dim];
}

// Batch GEMV: identical 128-lane summation/reduction and BF16 rounding
// as llm_gemv_bf16, preserving each token's mathematical independence.
// NDRange workgroup [row,token], local [128,1].
__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_prefill_gemv(
    __global const ushort *weights,
    __global const ushort *input,
    __global ushort *output,
    const uint in_dim,
    const uint out_dim,
    const uint batch_tokens
) {
    const uint row=(uint)get_group_id(0);
    const uint token=(uint)get_group_id(1);
    const uint tid=(uint)get_local_id(0);
    if(row>=out_dim || token>=batch_tokens) return;
    const uint weight_offset=row*in_dim;
    const uint input_offset=token*in_dim;
    float sum=0.0f;
    for(uint d=tid;d<in_dim;d+=128u) {
        sum=fma(read_bf16(weights[weight_offset+d]),
                read_bf16(input[input_offset+d]),sum);
    }
    __local float partials[128];
    partials[tid]=sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=64u;stride>0u;stride>>=1u){
        if(tid<stride) partials[tid]+=partials[tid+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(tid==0u) output[token*out_dim+row]=write_bf16(partials[0]);
}

__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_prefill_norm(
    __global const ushort *input,
    __global const ushort *weights,
    __global ushort *output,
    const uint hidden,
    const uint tokens,
    const float epsilon
) {
    const uint token=(uint)get_group_id(0);
    const uint tid=(uint)get_local_id(0);
    if(token>=tokens)return;
    const uint offset=token*hidden;
    __local float partials[128];
    float sum=0.0f;
    for(uint d=tid;d<hidden;d+=128u){
        const float x=read_bf16(input[offset+d]);
        sum=fma(x,x,sum);
    }
    partials[tid]=sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=64u;stride>0u;stride>>=1u){
        if(tid<stride)partials[tid]+=partials[tid+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float factor=rsqrt(partials[0]/(float)hidden+epsilon);
    for(uint d=tid;d<hidden;d+=128u){
        const ushort normalized=write_bf16(read_bf16(input[offset+d])*factor);
        output[offset+d]=write_bf16(read_bf16(normalized)*read_bf16(weights[d]));
    }
}

// Exactly the same rotate_half, BF16 cos/sin and arithmetic as the
// llm_rope token kernel, with an absolute position per input row.
__kernel void llm_prefill_rope(
    __global const ushort *input,
    __global ushort *output,
    const uint tokens,
    const uint num_heads,
    const uint dim,
    const uint start_position,
    const float theta,
    const uint llama3_scaling,
    const float scaling_factor,
    const float low_freq_factor,
    const float high_freq_factor,
    const float original_max_position
) {
    const uint i=(uint)get_global_id(0);
    const uint elements=num_heads*dim;
    if(i>=tokens*elements)return;
    const uint token=i/elements;
    const uint in_token=i%elements;
    const uint head=in_token/dim;
    const uint d=in_token%dim;
    const uint half_dim=dim/2u;
    const uint counterpart=token*elements+head*dim+(d<half_dim?d+half_dim:d-half_dim);
    const float rotated=d<half_dim?-read_bf16(input[counterpart]):read_bf16(input[counterpart]);
    const uint frequency_dim=(d%half_dim)*2u;
    float inverse=exp(-(float)frequency_dim*log(theta)/(float)dim);
    if(llama3_scaling!=0u){
        const float wavelength=6.28318530717958647692f/inverse;
        const float low_wavelength=original_max_position/low_freq_factor;
        const float high_wavelength=original_max_position/high_freq_factor;
        if(wavelength>low_wavelength){
            inverse/=scaling_factor;
        }else if(wavelength>=high_wavelength){
            const float smooth=(original_max_position/wavelength-low_freq_factor)/
                (high_freq_factor-low_freq_factor);
            inverse=(1.0f-smooth)*(inverse/scaling_factor)+smooth*inverse;
        }
    }
    const float angle=(float)(start_position+token)*inverse;
    const float c=read_bf16(write_bf16(cos(angle)));
    const float s=read_bf16(write_bf16(sin(angle)));
    const float left=read_bf16(write_bf16(read_bf16(input[i])*c));
    const float right=read_bf16(write_bf16(rotated*s));
    output[i]=write_bf16(left+right);
}

// Score matrix [batch_tokens,q_heads,history_tokens] where history_tokens
// is the end of this batch (previous context plus current batch). Only
// keys <= absolute query position are read. The remaining scores are
// initialized to zero but ignored by causal softmax/PV.
__kernel void llm_prefill_qk(
    __global const ushort *queries,
    __global const ushort *keys,
    __global float *scores,
    const uint batch_tokens,
    const uint preceding_tokens,
    const uint history_tokens,
    const uint q_heads,
    const uint kv_heads,
    const uint dim
) {
    const uint idx=(uint)get_global_id(0);
    const uint full=history_tokens*q_heads*batch_tokens;
    if(idx>=full)return;
    const uint token=idx/(history_tokens*q_heads);
    const uint inside=idx%(history_tokens*q_heads);
    const uint qh=inside/history_tokens;
    const uint keypos=inside%history_tokens;
    if(keypos>preceding_tokens+token){
        scores[idx]=0.0f;
        return;
    }
    const uint serving=qh/(q_heads/kv_heads);
    const uint qo=(token*q_heads+qh)*dim;
    const uint ko=(keypos*kv_heads+serving)*dim;
    float dot=0.0f;
    for(uint d=0u;d<dim;d++)
        dot=fma(read_bf16(queries[qo+d]),read_bf16(keys[ko+d]),dot);
    scores[idx]=dot*rsqrt((float)dim);
}

__kernel void llm_prefill_softmax(
    __global float *scores,
    const uint batch_tokens,
    const uint preceding_tokens,
    const uint history_tokens,
    const uint q_heads
){
    const uint idx=(uint)get_global_id(0);
    if(idx>=batch_tokens*q_heads)return;
    const uint token=idx/q_heads;
    const uint valid=preceding_tokens+token+1u;
    const uint start=idx*history_tokens;
    float maxval=-INFINITY;
    for(uint k=0u;k<valid;k++)
        maxval=fmax(maxval,scores[start+k]);
    float denom=0.0f;
    for(uint k=0u;k<valid;k++){
        const float p=exp(scores[start+k]-maxval);
        scores[start+k]=p;denom+=p;
    }
    for(uint k=0u;k<valid;k++)scores[start+k]/=denom;
}

__kernel void llm_prefill_pv(
    __global const float *probs,
    __global const ushort *values,
    __global ushort *output,
    const uint batch_tokens,
    const uint preceding_tokens,
    const uint history_tokens,
    const uint q_heads,
    const uint kv_heads,
    const uint dim
){
    const uint idx=(uint)get_global_id(0);
    if(idx>=batch_tokens*q_heads*dim)return;
    const uint token=idx/(q_heads*dim);
    const uint inner=idx%(q_heads*dim);
    const uint qh=inner/dim;
    const uint d=inner%dim;
    const uint serving=qh/(q_heads/kv_heads);
    const uint score_offset=(token*q_heads+qh)*history_tokens;
    float sum=0.0f;
    for(uint k=0u;k<=preceding_tokens+token;k++){
        const uint vo=(k*kv_heads+serving)*dim+d;
        sum=fma(probs[score_offset+k],read_bf16(values[vo]),sum);
    }
    output[idx]=write_bf16(sum);
}



// Weight-reuse BF16 matrix multiply: one GPU workgroup computes one
// matrix row for 16 neighboring prompt tokens, loading each BF16 weight
// only ONCE across those tokens. This preserves the per-thread fma
// accumulation order and the original 128-lane binary tree reduction
// for each output, so GPU BF16 logits remain bit-exact.
// Different from the old per-(row,token) launch (no weight reuse).
#define TERRSVUSION_WEIGHT_REUSE_TOKENS 16u
#define TERRSVUSION_WEIGHT_REUSE_TOKENS8 8u
__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_prefill_gemv_reuse8(
    __global const ushort *weights,
    __global const ushort *inputs,
    __global ushort *outputs,
    const uint in_dim,
    const uint out_dim,
    const uint tokens
) {
    const uint row=(uint)get_group_id(0);
    const uint tile=(uint)get_group_id(1);
    const uint tid=(uint)get_local_id(0);
    const uint first=tile*TERRSVUSION_WEIGHT_REUSE_TOKENS8;
    if (row>=out_dim || first>=tokens) return;
    const uint length=min((uint)TERRSVUSION_WEIGHT_REUSE_TOKENS8,tokens-first);
    float acc[8];
    for(uint i=0u;i<TERRSVUSION_WEIGHT_REUSE_TOKENS8;i++) acc[i]=0.0f;
    const uint woffset=row*in_dim;
    for(uint d=tid;d<in_dim;d+=128u) {
        const float w=read_bf16(weights[woffset+d]);
        for(uint token=0u;token<length;token++) {
            const float x=read_bf16(inputs[(first+token)*in_dim+d]);
            acc[token]=fma(w,x,acc[token]);
        }
    }
    __local float partials[8][128];
    for(uint token=0u;token<length;token++)
        partials[token][tid]=acc[token];
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=64u;stride>0u;stride>>=1u){
        if(tid<stride){
            for(uint token=0u;token<length;token++)
                partials[token][tid]+=partials[token][tid+stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(tid==0u){
        for(uint token=0u;token<length;token++){
            outputs[(first+token)*out_dim+row]=write_bf16(partials[token][0]);
        }
    }
}


__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_prefill_gemv_reuse16(
    __global const ushort *weights,
    __global const ushort *inputs,
    __global ushort *outputs,
    const uint in_dim,
    const uint out_dim,
    const uint tokens
) {
    const uint row=(uint)get_group_id(0);
    const uint tile=(uint)get_group_id(1);
    const uint tid=(uint)get_local_id(0);
    const uint first=tile*TERRSVUSION_WEIGHT_REUSE_TOKENS;
    if (row>=out_dim || first>=tokens) return;
    const uint length=min((uint)TERRSVUSION_WEIGHT_REUSE_TOKENS,tokens-first);
    float acc[16];
    for(uint i=0u;i<TERRSVUSION_WEIGHT_REUSE_TOKENS;i++) acc[i]=0.0f;
    const uint woffset=row*in_dim;
    for(uint d=tid;d<in_dim;d+=128u) {
        const float w=read_bf16(weights[woffset+d]);
        for(uint token=0u;token<length;token++) {
            const float x=read_bf16(inputs[(first+token)*in_dim+d]);
            acc[token]=fma(w,x,acc[token]);
        }
    }
    __local float partials[16][128];
    for(uint token=0u;token<length;token++)
        partials[token][tid]=acc[token];
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=64u;stride>0u;stride>>=1u){
        if(tid<stride){
            for(uint token=0u;token<length;token++)
                partials[token][tid]+=partials[token][tid+stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(tid==0u){
        for(uint token=0u;token<length;token++){
            outputs[(first+token)*out_dim+row]=write_bf16(partials[token][0]);
        }
    }
}

// Same operation, but reuses the same BF16 row across 32 tokens. The
// larger private accumulator and scratch may reduce occupancy; benchmark
// on the actual Radeon instead of assuming it is faster.
#define TERRSVUSION_WEIGHT_REUSE_TOKENS32 32u
__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_prefill_gemv_reuse32(
    __global const ushort *weights,
    __global const ushort *inputs,
    __global ushort *outputs,
    const uint in_dim,
    const uint out_dim,
    const uint tokens
) {
    const uint row=(uint)get_group_id(0);
    const uint tile=(uint)get_group_id(1);
    const uint tid=(uint)get_local_id(0);
    const uint first=tile*TERRSVUSION_WEIGHT_REUSE_TOKENS32;
    if(row>=out_dim||first>=tokens)return;
    const uint length=min((uint)TERRSVUSION_WEIGHT_REUSE_TOKENS32,tokens-first);
    float acc[32];
    for(uint i=0u;i<32u;i++) acc[i]=0.0f;
    const uint woffset=row*in_dim;
    for(uint d=tid;d<in_dim;d+=128u){
        const float w=read_bf16(weights[woffset+d]);
        for(uint token=0u;token<length;token++){
            const float x=read_bf16(inputs[(first+token)*in_dim+d]);
            acc[token]=fma(w,x,acc[token]);
        }
    }
    __local float partials[32][128];
    for(uint token=0u;token<length;token++)partials[token][tid]=acc[token];
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=64u;stride>0u;stride>>=1u){
        if(tid<stride){
            for(uint token=0u;token<length;token++)
                partials[token][tid]+=partials[token][tid+stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(tid==0u){
        for(uint token=0u;token<length;token++)
            outputs[(first+token)*out_dim+row]=write_bf16(partials[token][0]);
    }
}



// Optimized exact-arithmetic single-token BF16 GEMV: 4 independent
// global loads are issued in a block, but each lane accumulates in
// precisely the same d, d+128, d+256, ... order as llm_gemv_bf16.
// This keeps 128-lane tree reduction and BF16 output rounding intact.
__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_gemv_bf16_unroll4(
    __global const ushort *weights,
    __global const ushort *x,
    __global ushort *result,
    const uint in_dim,
    const uint out_dim
) {
    const uint row=(uint)get_group_id(0);
    const uint tid=(uint)get_local_id(0);
    if(row>=out_dim)return;
    const uint weight_base=row*in_dim;
    float sum=0.0f;
    uint d=tid;
    for(;d+384u<in_dim;d+=512u){
        // As in the original llm_gemv_bf16: each work-item
        // accumulates positions tid+128*n in order.
        const float w0=read_bf16(weights[weight_base+d]);
        const float w1=read_bf16(weights[weight_base+d+128u]);
        const float w2=read_bf16(weights[weight_base+d+256u]);
        const float w3=read_bf16(weights[weight_base+d+384u]);
        const float x0=read_bf16(x[d]);
        const float x1=read_bf16(x[d+128u]);
        const float x2=read_bf16(x[d+256u]);
        const float x3=read_bf16(x[d+384u]);
        sum=fma(w0,x0,sum);
        sum=fma(w1,x1,sum);
        sum=fma(w2,x2,sum);
        sum=fma(w3,x3,sum);
    }
    for(;d<in_dim;d+=128u)
        sum=fma(read_bf16(weights[weight_base+d]),read_bf16(x[d]),sum);
    __local float partials[128];
    partials[tid]=sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=64u;stride>0u;stride>>=1u){
        if(tid<stride)partials[tid]+=partials[tid+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(tid==0u)result[row]=write_bf16(partials[0]);
}



// GPU decode fusion v1. Reuses original BF16 GEMV arithmetic with the
// SAME 128-lane accumulation/reduction order as three standalone Q/K/V
// kernels; just one OpenCL launch and one shared activation load per lane.
// No quantization, no change in RoPE, attention or canonical ColdStore.
// Per-row Q, K and V weights use original BF16 device allocations.
__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_fused_qkv_bf16(
    __global const ushort *q_weights,
    __global const ushort *k_weights,
    __global const ushort *v_weights,
    __global const ushort *input,
    __global ushort *out_q,
    __global ushort *out_k,
    __global ushort *out_v,
    const uint hidden,
    const uint q_rows,
    const uint kv_rows
) {
    const uint row=(uint)get_group_id(0);
    const uint lane=(uint)get_local_id(0);
    if(row>=q_rows)return;
    const uint offset=row*hidden;
    const bool kv_valid=row<kv_rows;
    float q=0.0f;
    float k=0.0f;
    float v=0.0f;
    for(uint dim=lane;dim<hidden;dim+=128u) {
        const float x=read_bf16(input[dim]);
        q=fma(read_bf16(q_weights[offset+dim]),x,q);
        if(kv_valid) {
            k=fma(read_bf16(k_weights[offset+dim]),x,k);
            v=fma(read_bf16(v_weights[offset+dim]),x,v);
        }
    }
    __local float partial_q[128];
    __local float partial_k[128];
    __local float partial_v[128];
    partial_q[lane]=q;
    partial_k[lane]=k;
    partial_v[lane]=v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=64u;stride>0u;stride>>=1u){
        if(lane<stride){
            partial_q[lane]+=partial_q[lane+stride];
            partial_k[lane]+=partial_k[lane+stride];
            partial_v[lane]+=partial_v[lane+stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0u){
        out_q[row]=write_bf16(partial_q[0]);
        if(kv_valid){
            out_k[row]=write_bf16(partial_k[0]);
            out_v[row]=write_bf16(partial_v[0]);
        }
    }
}

// MLP gate and up share the normalized BF16 input. A single 128-lane
// workgroup performs BOTH original BF16 GEMVs and the SAME BF16 SwiGLU
// rounding sequence. Removes 3 kernel launches -> 1. No hidden state
// of the original model or its BF16 numerical storage is discarded.
__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_fused_swiglu_bf16(
    __global const ushort *gate_weights,
    __global const ushort *up_weights,
    __global const ushort *normalized,
    __global ushort *activation,
    const uint hidden,
    const uint intermediate
) {
    const uint row=(uint)get_group_id(0);
    const uint lane=(uint)get_local_id(0);
    if(row>=intermediate)return;
    const uint offset=row*hidden;
    float g=0.0f, u=0.0f;
    for(uint d=lane;d<hidden;d+=128u) {
        const float x=read_bf16(normalized[d]);
        g=fma(read_bf16(gate_weights[offset+d]),x,g);
        u=fma(read_bf16(up_weights[offset+d]),x,u);
    }
    __local float gpartial[128];
    __local float upartial[128];
    gpartial[lane]=g;upartial[lane]=u;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=64u;stride>0u;stride>>=1u){
        if(lane<stride){
            gpartial[lane]+=gpartial[lane+stride];
            upartial[lane]+=upartial[lane+stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0u){
        const float gate=read_bf16(write_bf16(gpartial[0]));
        const float up=read_bf16(write_bf16(upartial[0]));
        const float silu=gate/(1.0f+exp(-gate));
        const ushort gate_activated=write_bf16(silu);
        activation[row]=write_bf16(read_bf16(gate_activated)*up);
    }
}



// OPTIONAL positive-epsilon canonical MICRO-REGION sparse attention.
// Micro-metadata links and BF16 min/max boxes are resolved on CPU using
// the *original* dimensional axis and an omitted-mass certificate.
// Only positions certified as retained are submitted here. This kernel
// never uses 4-bit approximations for actual attention calculations.
//
// We preserve the original dense GPU Q·K FMA sequence for each selected
// score. Every omitted score is explicitly -INFINITY, so the original
// softmax/PV kernels normalize only the certified retained token set.
__kernel void llm_micro_clear_scores(
    __global float *scores,
    const uint score_count
) {
    const uint idx=(uint)get_global_id(0);
    if(idx<score_count)scores[idx]=-INFINITY;
}

__kernel void llm_micro_sparse_qk(
    __global const ushort *query,
    __global const ushort *key_cache,
    __global const uint *token_indices,
    __global float *scores,
    const uint selected_tokens,
    const uint full_tokens,
    const uint q_heads,
    const uint kv_heads,
    const uint dim
) {
    const uint i=(uint)get_global_id(0);
    if(i>=selected_tokens*q_heads)return;
    const uint qh=i/selected_tokens;
    const uint selected=i%selected_tokens;
    const uint token=token_indices[selected];
    if(token>=full_tokens)return;
    const uint kvh=qh/(q_heads/kv_heads);
    const uint ko=(token*kv_heads+kvh)*dim;
    const uint qo=qh*dim;
    float dot=0.0f;
    for(uint d=0u;d<dim;d++)
        dot=fma(read_bf16(query[qo+d]),read_bf16(key_cache[ko+d]),dot);
    scores[qh*full_tokens+token]=dot*rsqrt((float)dim);
}



// -------------------------------------------------------------------------
// OPT-IN MICRO-METADATA BF16 RESIDENT ATTENTION (GPU Phoenix).
//
// One OpenCL kernel, no BF16 V transfer or cold read per query.
// Indices are preuploaded and remain in GPU memory until the certified
// selection changes. Exact BF16 keys and values are always read from
// the original GPU KV buffers; the canonical CPU ColdStore is separately
// committed/verifiable and remains the persistence authority.
//
// This kernel is ONLY for explicitly positive tolerance and a
// CPU-certified conservative bound on omitted softmax probability mass.
// It cannot be used for unvalidated 4-bit/LSH proposals.
// All original BF16 QK multiplications (for selected K) and value output
// BF16 rounding are preserved; discarded positions are re-normalized.
// -------------------------------------------------------------------------
#define TV_MICRO_HOT_MAX_SELECTED 2048u
__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_micro_hot_fused_attention(
    __global const ushort *query,
    __global const ushort *key_cache,
    __global const ushort *value_cache,
    __global const uint *selected_indices,
    __global ushort *output,
    const uint selected_count,
    const uint committed_tokens,
    const uint q_heads,
    const uint kv_heads,
    const uint dim
) {
    const uint qh=(uint)get_group_id(0);
    const uint lane=(uint)get_local_id(0);
    if(qh>=q_heads||selected_count==0u
       ||selected_count>TV_MICRO_HOT_MAX_SELECTED)return;
    const uint serving=qh/(q_heads/kv_heads);
    const uint q_offset=qh*dim;
    __local float scores[TV_MICRO_HOT_MAX_SELECTED];
    __local float total_mass;
    // Each BF16 dot uses exactly the original sequential fma order,
    // FP32 rsqrt and the GPU-native FP32 exp.
    for(uint idx=lane;idx<selected_count;idx+=128u){
        const uint pos=selected_indices[idx];
        float result=-INFINITY;
        if(pos<committed_tokens){
            const uint key_offset=(pos*kv_heads+serving)*dim;
            float dot=0.0f;
            for(uint d=0u;d<dim;d++){
                dot=fma(read_bf16(query[q_offset+d]),
                        read_bf16(key_cache[key_offset+d]),dot);
            }
            result=dot*rsqrt((float)dim);
        }
        scores[idx]=result;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lane==0u){
        float maxv=-INFINITY;
        for(uint t=0u;t<selected_count;t++)
            maxv=fmax(maxv,scores[t]);
        float sum=0.0f;
        for(uint t=0u;t<selected_count;t++){
            scores[t]=exp(scores[t]-maxv);
            sum+=scores[t];
        }
        total_mass=sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const float denominator=total_mass;
    for(uint d=lane;d<dim;d+=128u){
        float acc=0.0f;
        for(uint idx=0u;idx<selected_count;idx++){
            const uint pos=selected_indices[idx];
            if(pos<committed_tokens){
                const uint v_offset=(pos*kv_heads+serving)*dim+d;
                const float weight=scores[idx]/denominator;
                acc=fma(weight,read_bf16(value_cache[v_offset]),acc);
            }
        }
        output[q_offset+d]=write_bf16(acc);
    }
}



// ----------------------------------------------------------------------
// EXPERIMENTAL EXACT BF16 GPU-RESIDENT ATTENTION FUSION
// For contexts <=2048, one kernel replaces separate QK+softmax+PV
// launches while evaluating EVERY BF16 K and V token, in original
// per-query-head sequential QK/PV FMA order. No indices are used and
// NO memory or probability mass is discarded. Original BF16 ColdStore,
// CPU branch-and-bound, 4-bit micro metadata remain untouched.
// Always opt-in until full GPU BF16 logits and KV parity are verified.
// ----------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_exact_hot_fused_attention(
    __global const ushort *query,
    __global const ushort *key_cache,
    __global const ushort *value_cache,
    __global ushort *output,
    const uint tokens,
    const uint q_heads,
    const uint kv_heads,
    const uint dim
) {
    const uint head=(uint)get_group_id(0);
    const uint lane=(uint)get_local_id(0);
    if(head>=q_heads || tokens==0u || tokens>2048u)return;
    const uint serving=head/(q_heads/kv_heads);
    const uint qbase=head*dim;
    __local float scores[2048];
    __local float denominator;
    for(uint token=lane;token<tokens;token+=128u){
        const uint keybase=(token*kv_heads+serving)*dim;
        float dot=0.0f;
        for(uint d=0u;d<dim;d++)
            dot=fma(read_bf16(query[qbase+d]),
                    read_bf16(key_cache[keybase+d]),dot);
        scores[token]=dot*rsqrt((float)dim);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lane==0u){
        float maximum=-INFINITY;
        for(uint token=0u;token<tokens;token++)
            maximum=fmax(maximum,scores[token]);
        float sum=0.0f;
        for(uint token=0u;token<tokens;token++){
            const float probability=exp(scores[token]-maximum);
            scores[token]=probability;
            sum+=probability;
        }
        denominator=sum;
        for(uint token=0u;token<tokens;token++)
            scores[token]/=sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint d=lane;d<dim;d+=128u){
        float result=0.0f;
        for(uint token=0u;token<tokens;token++){
            const uint offset=(token*kv_heads+serving)*dim+d;
            result=fma(scores[token],read_bf16(value_cache[offset]),result);
        }
        output[qbase+d]=write_bf16(result);
    }
}


// GPU-only optional register-resident BF16 prefill GEMV variants.
// Identical per-lane BF16->FP32 FMA accumulation and 128-lane
// binary reduction as the proven original kernels. Dynamic
// token loops are replaced by compile-time unrolled token
// arrays to keep 8/16 accumulators in physical GPU registers.
// Every token is still computed, no truncation or quantization.

__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_prefill_gemv_reuse8_register(
    __global const ushort *weights,
    __global const ushort *inputs,
    __global ushort *outputs,
    const uint in_dim,
    const uint out_dim,
    const uint tokens
) {
    const uint row=(uint)get_group_id(0);
    const uint tile=(uint)get_group_id(1);
    const uint tid=(uint)get_local_id(0);
    const uint first=tile*8u;
    if(row>=out_dim||first>=tokens)return;
    const uint weight_offset=row*in_dim;
    float acc[8];
    #pragma unroll
    for(uint t=0u;t<8u;t++)acc[t]=0.0f;
    for(uint d=tid;d<in_dim;d+=128u){
        const float w=read_bf16(weights[weight_offset+d]);
        #pragma unroll
        for(uint t=0u;t<8u;t++){
            if(first+t<tokens){
                const float x=read_bf16(inputs[(first+t)*in_dim+d]);
                acc[t]=fma(w,x,acc[t]);
            }
        }
    }
    __local float partials[8][128];
    #pragma unroll
    for(uint t=0u;t<8u;t++)
        partials[t][tid]=acc[t];
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=64u;stride>0u;stride>>=1u){
        if(tid<stride){
            #pragma unroll
            for(uint t=0u;t<8u;t++)
                partials[t][tid]+=partials[t][tid+stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(tid==0u){
        #pragma unroll
        for(uint t=0u;t<8u;t++){
            if(first+t<tokens)
                outputs[(first+t)*out_dim+row]=write_bf16(partials[t][0]);
        }
    }
}

__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_prefill_gemv_reuse16_register(
    __global const ushort *weights,
    __global const ushort *inputs,
    __global ushort *outputs,
    const uint in_dim,
    const uint out_dim,
    const uint tokens
) {
    const uint row=(uint)get_group_id(0);
    const uint tile=(uint)get_group_id(1);
    const uint tid=(uint)get_local_id(0);
    const uint first=tile*16u;
    if(row>=out_dim||first>=tokens)return;
    const uint weight_offset=row*in_dim;
    float acc[16];
    #pragma unroll
    for(uint t=0u;t<16u;t++)acc[t]=0.0f;
    for(uint d=tid;d<in_dim;d+=128u){
        const float w=read_bf16(weights[weight_offset+d]);
        #pragma unroll
        for(uint t=0u;t<16u;t++){
            if(first+t<tokens){
                const float x=read_bf16(inputs[(first+t)*in_dim+d]);
                acc[t]=fma(w,x,acc[t]);
            }
        }
    }
    __local float partials[16][128];
    #pragma unroll
    for(uint t=0u;t<16u;t++)
        partials[t][tid]=acc[t];
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=64u;stride>0u;stride>>=1u){
        if(tid<stride){
            #pragma unroll
            for(uint t=0u;t<16u;t++)
                partials[t][tid]+=partials[t][tid+stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(tid==0u){
        #pragma unroll
        for(uint t=0u;t<16u;t++){
            if(first+t<tokens)
                outputs[(first+t)*out_dim+row]=write_bf16(partials[t][0]);
        }
    }
}

// OpenCL AMD GFX1103 experimental virtual-lane BF16 GEMV.
// Each physical work-item computes 2 or 4 of the EXACT original
// 128 independent FP32 FMA chains in original d,d+128... order.
// Then combines them in precisely the original binary tree:
// stride64 before stride32 before stride16..1. These kernels
// preserve BF16 weights, KV, and every token; only the physical
// wavefront shape / inter-wave barriers are changed.


__kernel __attribute__((reqd_work_group_size(64,1,1)))
void llm_prefill_gemv_reuse8_virtual64(
    __global const ushort *weights,
    __global const ushort *inputs,
    __global ushort *outputs,
    const uint in_dim,
    const uint out_dim,
    const uint tokens
) {
    const uint row=(uint)get_group_id(0);
    const uint tile=(uint)get_group_id(1);
    const uint lane=(uint)get_local_id(0);
    const uint first=tile*8u;
    if(row>=out_dim||first>=tokens)return;
    const uint length=min(8u,tokens-first);
    const uint weight_base=row*in_dim;
    float acc0[8]; float acc1[8];
    for(uint t=0u;t<8u;t++) {
        acc0[t]=0.0f; acc1[t]=0.0f;
    }
    for(uint d=lane;d<in_dim;d+=128u){
        const uint valid0=(d<in_dim);
        const float w0=valid0?read_bf16(weights[weight_base+d]):0.0f;
        const uint valid1=(d+64u<in_dim);
        const float w1=valid1?read_bf16(weights[weight_base+d+64u]):0.0f;
        for(uint t=0u;t<length;t++){
            if(valid0) acc0[t]=fma(w0,read_bf16(inputs[(first+t)*in_dim+d]),acc0[t]);
            if(valid1) acc1[t]=fma(w1,read_bf16(inputs[(first+t)*in_dim+d+64u]),acc1[t]);
        }
    }
    __local float partials[8][64];
    for(uint t=0u;t<length;t++)
        partials[t][lane]=acc0[t]+acc1[t];
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=32u;stride>0u;stride>>=1u){
        if(lane<stride){
            for(uint t=0u;t<length;t++)
                partials[t][lane]+=partials[t][lane+stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0u){
        for(uint t=0u;t<length;t++)
            outputs[(first+t)*out_dim+row]=write_bf16(partials[t][0]);
    }
}


__kernel __attribute__((reqd_work_group_size(32,1,1)))
void llm_prefill_gemv_reuse8_virtual32(
    __global const ushort *weights,
    __global const ushort *inputs,
    __global ushort *outputs,
    const uint in_dim,
    const uint out_dim,
    const uint tokens
) {
    const uint row=(uint)get_group_id(0);
    const uint tile=(uint)get_group_id(1);
    const uint lane=(uint)get_local_id(0);
    const uint first=tile*8u;
    if(row>=out_dim||first>=tokens)return;
    const uint length=min(8u,tokens-first);
    const uint weight_base=row*in_dim;
    float acc0[8]; float acc1[8]; float acc2[8]; float acc3[8];
    for(uint t=0u;t<8u;t++) {
        acc0[t]=0.0f; acc1[t]=0.0f; acc2[t]=0.0f; acc3[t]=0.0f;
    }
    for(uint d=lane;d<in_dim;d+=128u){
        const uint valid0=(d<in_dim);
        const float w0=valid0?read_bf16(weights[weight_base+d]):0.0f;
        const uint valid1=(d+32u<in_dim);
        const float w1=valid1?read_bf16(weights[weight_base+d+32u]):0.0f;
        const uint valid2=(d+64u<in_dim);
        const float w2=valid2?read_bf16(weights[weight_base+d+64u]):0.0f;
        const uint valid3=(d+96u<in_dim);
        const float w3=valid3?read_bf16(weights[weight_base+d+96u]):0.0f;
        for(uint t=0u;t<length;t++){
            if(valid0) acc0[t]=fma(w0,read_bf16(inputs[(first+t)*in_dim+d]),acc0[t]);
            if(valid1) acc1[t]=fma(w1,read_bf16(inputs[(first+t)*in_dim+d+32u]),acc1[t]);
            if(valid2) acc2[t]=fma(w2,read_bf16(inputs[(first+t)*in_dim+d+64u]),acc2[t]);
            if(valid3) acc3[t]=fma(w3,read_bf16(inputs[(first+t)*in_dim+d+96u]),acc3[t]);
        }
    }
    __local float partials[8][32];
    for(uint t=0u;t<length;t++)
        partials[t][lane]=(acc0[t]+acc2[t])+(acc1[t]+acc3[t]);
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=16u;stride>0u;stride>>=1u){
        if(lane<stride){
            for(uint t=0u;t<length;t++)
                partials[t][lane]+=partials[t][lane+stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0u){
        for(uint t=0u;t<length;t++)
            outputs[(first+t)*out_dim+row]=write_bf16(partials[t][0]);
    }
}


// -------------------------------------------------------------------------
// EXPERIMENTAL GPU PREFILL BF16 WEIGHT-REUSE8 PROJECTION FUSION (opt-in)
// Q, K, V share the exact normalized BF16 activation, with each of the
// three projections retaining the ORIGINAL 128-lane per-row BF16->FP32
// FMA order and binary-tree reduction; 1 GPU launch replaces 3.
// Every original BF16 Q/K/V output is stored (no approximation/skipping).
// -------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_prefill_fused_qkv_reuse8(
    __global const ushort *q_weights,
    __global const ushort *k_weights,
    __global const ushort *v_weights,
    __global const ushort *input,
    __global ushort *output_q,
    __global ushort *output_k,
    __global ushort *output_v,
    const uint hidden,
    const uint q_rows,
    const uint kv_rows,
    const uint tokens
) {
    const uint row=(uint)get_group_id(0);
    const uint tile=(uint)get_group_id(1);
    const uint lane=(uint)get_local_id(0);
    const uint first=tile*8u;
    if(row>=q_rows||first>=tokens)return;
    const uint len=min(8u,tokens-first);
    const bool is_kv=row<kv_rows;
    const uint off=row*hidden;
    float q[8],k[8],v[8];
    for(uint i=0u;i<8u;i++){q[i]=0.0f;k[i]=0.0f;v[i]=0.0f;}
    for(uint d=lane;d<hidden;d+=128u){
        const float wq=read_bf16(q_weights[off+d]);
        const float wk=is_kv?read_bf16(k_weights[off+d]):0.0f;
        const float wv=is_kv?read_bf16(v_weights[off+d]):0.0f;
        for(uint token=0u;token<len;token++){
            const float x=read_bf16(input[(first+token)*hidden+d]);
            q[token]=fma(wq,x,q[token]);
            if(is_kv){
                k[token]=fma(wk,x,k[token]);
                v[token]=fma(wv,x,v[token]);
            }
        }
    }
    __local float local_q[8][128];
    __local float local_k[8][128];
    __local float local_v[8][128];
    for(uint t=0u;t<len;t++){
        local_q[t][lane]=q[t];
        if(is_kv){
            local_k[t][lane]=k[t];
            local_v[t][lane]=v[t];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=64u;stride>0u;stride>>=1u){
        if(lane<stride){
            for(uint t=0u;t<len;t++){
                local_q[t][lane]+=local_q[t][lane+stride];
                if(is_kv){
                    local_k[t][lane]+=local_k[t][lane+stride];
                    local_v[t][lane]+=local_v[t][lane+stride];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0u){
        for(uint t=0u;t<len;t++){
            output_q[(first+t)*q_rows+row]=write_bf16(local_q[t][0]);
            if(is_kv){
                output_k[(first+t)*kv_rows+row]=write_bf16(local_k[t][0]);
                output_v[(first+t)*kv_rows+row]=write_bf16(local_v[t][0]);
            }
        }
    }
}

// One launch for gate/up GEMV AND original BF16 SwiGLU, reusing the
// same normalized BF16 input and original gate/up weight bytes.
// Both GEMV reductions use the same 128-lane + 7-stage tree, then
// separately ROUND gate and up to BF16 before SiLU and product, exactly
// as llm_prefill_gemv_reuse8 + llm_swiglu.
__kernel __attribute__((reqd_work_group_size(128,1,1)))
void llm_prefill_fused_swiglu_reuse8(
    __global const ushort *gate_weights,
    __global const ushort *up_weights,
    __global const ushort *normalized,
    __global ushort *activation,
    const uint hidden,
    const uint intermediate,
    const uint tokens
) {
    const uint row=(uint)get_group_id(0);
    const uint tile=(uint)get_group_id(1);
    const uint lane=(uint)get_local_id(0);
    const uint first=tile*8u;
    if(row>=intermediate||first>=tokens)return;
    const uint len=min(8u,tokens-first);
    const uint off=row*hidden;
    float gate[8],up[8];
    for(uint i=0u;i<8u;i++){gate[i]=0.0f;up[i]=0.0f;}
    for(uint d=lane;d<hidden;d+=128u){
        const float wg=read_bf16(gate_weights[off+d]);
        const float wu=read_bf16(up_weights[off+d]);
        for(uint t=0u;t<len;t++){
            const float x=read_bf16(normalized[(first+t)*hidden+d]);
            gate[t]=fma(wg,x,gate[t]);
            up[t]=fma(wu,x,up[t]);
        }
    }
    __local float local_gate[8][128];
    __local float local_up[8][128];
    for(uint t=0u;t<len;t++){
        local_gate[t][lane]=gate[t];
        local_up[t][lane]=up[t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=64u;stride>0u;stride>>=1u){
        if(lane<stride){
            for(uint t=0u;t<len;t++){
                local_gate[t][lane]+=local_gate[t][lane+stride];
                local_up[t][lane]+=local_up[t][lane+stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0u){
        for(uint t=0u;t<len;t++){
            const float g=read_bf16(write_bf16(local_gate[t][0]));
            const float u=read_bf16(write_bf16(local_up[t][0]));
            const float activated=g/(1.0f+exp(-g));
            const ushort gate_bf16=write_bf16(activated);
            activation[(first+t)*intermediate+row]=
                write_bf16(read_bf16(gate_bf16)*u);
        }
    }
}
