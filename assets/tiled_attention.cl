// Included after gpu_llm.cl so original BF16 conversion is shared, not copied.
// OpenCL C 1.2; no fast-math, subgroup or native BF16 requirement.
// AT_D, AT_BM, AT_BN, AT_WG are part of the compiled-program identity.
// Strict mode recomputes QK three times to preserve the reference operation order.
// Online/split modes are explicit experiments, NOT bit-equivalent contracts.
#pragma OPENCL FP_CONTRACT OFF
#define AT_R ((AT_D + AT_WG - 1) / AT_WG)

__kernel __attribute__((reqd_work_group_size(AT_WG,1,1)))
void mslim_attention_tiled(
    __global const ushort *q, __global const ushort *k, __global const ushort *v,
    __global ushort *output, __global float *partials,
    const uint queries, const uint prior, const uint history,
    const uint qheads, const uint kvheads, const uint dim,
    const uint online, const uint split_count, const uint first_group
) {
    const uint lane = (uint)get_local_id(0);
    const uint row_group = first_group + (uint)get_group_id(0);
    const uint qh = row_group % qheads;
    const uint first_query = (row_group / qheads) * AT_BM;
    const uint split = (uint)get_group_id(1);
    const uint kvh = qh / (qheads / kvheads);
    const uint chunk = (history + split_count - 1u) / split_count;
    const uint begin = split * chunk;
    const uint end = min(history, begin + chunk);
    __local ushort query_tile[AT_BM * AT_D];
    // Same storage reused for K then V, so local memory does not double.
    __local ushort data_tile[AT_BN * AT_D];
    __local float score[AT_BM * AT_BN];
    __local float maximum[AT_BM], denominator[AT_BM], alpha[AT_BM], spare[AT_BM];
    float accum[AT_BM][AT_R];
    for (uint r = 0; r < AT_BM; ++r) {
        for (uint x = 0; x < AT_R; ++x) accum[r][x] = 0.0f;
    }
    for (uint x = lane; x < AT_BM * AT_D; x += AT_WG) {
        uint row = first_query + x / AT_D;
        query_tile[x] = row < queries ? q[((size_t)row*qheads + qh)*dim + x%AT_D] : (ushort)0;
    }
    if (lane == 0) {
        for (uint r = 0; r < AT_BM; ++r) { maximum[r] = -INFINITY; denominator[r] = 0.0f; }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint passes = online ? 1u : 3u;
    for (uint pass = 0; pass < passes; ++pass) {
        for (uint base = begin; base < end; base += AT_BN) {
            const uint count = min((uint)AT_BN, end-base);
            for (uint x = lane; x < count*dim; x += AT_WG) {
                uint token = base + x/dim;
                data_tile[x] = k[((size_t)token*kvheads+kvh)*dim + x%dim];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint r = 0; r < AT_BM; ++r) {
                const uint row = first_query+r;
                for (uint t = lane; t < count; t += AT_WG) {
                    float s = -INFINITY;
                    if (row < queries && base+t <= prior+row) {
                        float dot = 0.0f;
                        // Same sequential FMA and runtime rsqrt as original QK.
                        for (uint d = 0; d < dim; ++d)
                            dot = fma(read_bf16(query_tile[r*AT_D+d]), read_bf16(data_tile[t*AT_D+d]), dot);
                        s = dot * rsqrt((float)dim);
                    }
                    score[r*AT_BN+t] = s;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane == 0) {
                for (uint r = 0; r < AT_BM; ++r) {
                    uint row = first_query+r;
                    uint valid = row < queries && prior+row >= base ? min(count, prior+row-base+1u) : 0u;
                    if (online) {
                        float old_m = maximum[r];
                        float new_m = old_m;
                        for (uint t=0; t<valid; ++t) new_m = fmax(new_m,score[r*AT_BN+t]);
                        float a = denominator[r] == 0.0f ? 0.0f : exp(old_m-new_m);
                        float sum = 0.0f;
                        for (uint t=0; t<valid; ++t) { float p=exp(score[r*AT_BN+t]-new_m); score[r*AT_BN+t]=p; sum+=p; }
                        maximum[r]=new_m; denominator[r]=denominator[r]*a+sum; alpha[r]=a;
                    } else if (pass == 0) {
                        for (uint t=0; t<valid; ++t) maximum[r]=fmax(maximum[r],score[r*AT_BN+t]);
                    } else if (pass == 1) {
                        float sum=denominator[r];
                        for (uint t=0; t<valid; ++t) sum+=exp(score[r*AT_BN+t]-maximum[r]);
                        denominator[r]=sum;
                    } else {
                        for (uint t=0; t<valid; ++t) score[r*AT_BN+t]=exp(score[r*AT_BN+t]-maximum[r])/denominator[r];
                    }
                    spare[r]=(float)valid;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (online || pass == 2) {
                for (uint x=lane; x<count*dim; x+=AT_WG) {
                    uint token=base+x/dim;
                    data_tile[x]=v[((size_t)token*kvheads+kvh)*dim+x%dim];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                for (uint r=0; r<AT_BM; ++r) {
                    for (uint x=0; x<AT_R; ++x) {
                        uint d=lane+x*AT_WG;
                        if (d<dim) {
                            float a=accum[r][x];
                            if (online) a*=alpha[r];
                            for (uint t=0; t<(uint)spare[r]; ++t)
                                a=fma(score[r*AT_BN+t],read_bf16(data_tile[t*AT_D+d]),a);
                            accum[r][x]=a;
                        }
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    for (uint r=0; r<AT_BM; ++r) {
        uint row=first_query+r;
        if (row < queries) {
            size_t partial_base=(((size_t)row*qheads+qh)*split_count+split)*(dim+2u);
            for (uint x=0; x<AT_R; ++x) {
                uint d=lane+x*AT_WG;
                if (d<dim) {
                    float a=accum[r][x];
                    if (split_count>1) partials[partial_base+2u+d]=a;
                    else output[((size_t)row*qheads+qh)*dim+d]=write_bf16(online ? a/denominator[r] : a);
                }
            }
            if (split_count>1 && lane==0) { partials[partial_base]=maximum[r]; partials[partial_base+1u]=denominator[r]; }
        }
    }
}

// Fixed split-order reduction; numerical equivalence, NOT reference bit identity.
__kernel __attribute__((reqd_work_group_size(AT_WG,1,1)))
void mslim_attention_merge(__global const float *partials, __global ushort *output,
    const uint qheads, const uint dim, const uint splits) {
    const uint qh=(uint)get_group_id(0), lane=(uint)get_local_id(0);
    __local float scale[64], denom;
    const size_t base=(size_t)qh*splits*(dim+2u);
    if (lane==0) {
        float m=-INFINITY;
        for(uint s=0;s<splits;++s) if(partials[base+s*(dim+2u)+1u]>0.0f) m=fmax(m,partials[base+s*(dim+2u)]);
        float l=0.0f;
        for(uint s=0;s<splits;++s) {
            float dl=partials[base+s*(dim+2u)+1u];
            scale[s]=dl>0.0f ? exp(partials[base+s*(dim+2u)]-m) : 0.0f;
            l=fma(scale[s],dl,l);
        }
        denom=l;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint d=lane;d<dim;d+=AT_WG) {
        float o=0.0f;
        for(uint s=0;s<splits;++s) o=fma(scale[s],partials[base+s*(dim+2u)+2u+d],o);
        output[(size_t)qh*dim+d]=write_bf16(o/denom);
    }
}
