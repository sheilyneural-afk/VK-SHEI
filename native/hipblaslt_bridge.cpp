// M-SLIM V3: optional BF16 matrix backend for real AMD gfx1103.
// C++ HIP kernel interface (the model orchestrator stays in Rust).
// Only native hipBLASLt gfx1103 is used; HSA_OVERRIDE_GFX_VERSION is refused.
// Original BF16 weight bytes are uploaded unchanged, no conversion to fp16.
// This module is optional, not linked into CPU or OpenCL-only executables.
// Exact values are rounded once BF16 after FP32 GEMM accumulation.
// GPL-independent AMD HIP/hipBLASLt runtime, loaded separately at runtime.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <map>
#include <tuple>
#include <memory>
#include <mutex>
#include <algorithm>
#include <new>

namespace {
thread_local std::string ERROR;
int fail(const char *name, int code) {
    ERROR=std::string(name)+" returned "+std::to_string(code);
    return code?code:-1;
}
int check_hip(hipError_t code,const char* what) {
    if(code==hipSuccess)return 0;
    ERROR=std::string(what)+": "+hipGetErrorString(code);
    return 100+(int)code;
}
int check_lt(hipblasStatus_t code,const char* what) {
    if(code==HIPBLAS_STATUS_SUCCESS)return 0;
    ERROR=std::string(what)+" hipBLASLt status="+std::to_string((int)code);
    return 200+(int)code;
}
struct Plan {
    hipblasLtMatmulDesc_t desc{};
    hipblasLtMatrixLayout_t A{},B{},C{};
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_size{};
    void* workspace{};
    ~Plan() {
        if(workspace)hipFree(workspace);
        if(A)hipblasLtMatrixLayoutDestroy(A);
        if(B)hipblasLtMatrixLayoutDestroy(B);
        if(C)hipblasLtMatrixLayoutDestroy(C);
        if(desc)hipblasLtMatmulDescDestroy(desc);
    }
};
struct Context {
    hipblasLtHandle_t handle{};
    void* input{};
    void* output{};
    size_t input_size{};
    size_t output_size{};
    std::map<std::tuple<int,int,int>,std::unique_ptr<Plan>> plans;
    std::mutex mutex;
    ~Context() {
        plans.clear();
        if(input)hipFree(input);
        if(output)hipFree(output);
        if(handle)hipblasLtDestroy(handle);
    }
};
bool compatible_device() {
    if(const char*override=std::getenv("HSA_OVERRIDE_GFX_VERSION")) {
        ERROR=std::string("HSA_OVERRIDE_GFX_VERSION=")+override+
              " is unsafe on gfx1103; native hipBLASLt requires no override";
        return false;
    }
    int count=0;
    if(check_hip(hipGetDeviceCount(&count),"hipGetDeviceCount")||count<1){
        if(count<1)ERROR="No HIP GPU device";
        return false;
    }
    hipDeviceProp_t properties{};
    if(check_hip(hipGetDeviceProperties(&properties,0),"hipGetDeviceProperties"))return false;
    if(std::string(properties.gcnArchName)!="gfx1103") {
        ERROR=std::string("This experimental bridge requires native gfx1103, found ")
              +properties.gcnArchName;
        return false;
    }
    return true;
}
bool grow(void** memory,size_t*capacity,size_t bytes,const char* what) {
    if(*capacity>=bytes)return true;
    if(*memory) {
        if(check_hip(hipFree(*memory),"hipFree scratch"))return false;
    }
    *memory=nullptr;*capacity=0;
    if(check_hip(hipMalloc(memory,bytes),what))return false;
    *capacity=bytes;
    return true;
}
Plan* make_plan(Context&context,int M,int N,int K) {
    const auto key=std::make_tuple(M,N,K);
    const auto found=context.plans.find(key);
    if(found!=context.plans.end())return found->second.get();
    auto plan=std::make_unique<Plan>();
    if(check_lt(hipblasLtMatmulDescCreate(&plan->desc,
          HIPBLAS_COMPUTE_32F,HIP_R_32F),"MatmulDescCreate"))return nullptr;
    hipblasOperation_t transA=HIPBLAS_OP_T,transB=HIPBLAS_OP_N;
    if(check_lt(hipblasLtMatmulDescSetAttribute(plan->desc,
          HIPBLASLT_MATMUL_DESC_TRANSA,&transA,sizeof(transA)),"Set transA"))return nullptr;
    if(check_lt(hipblasLtMatmulDescSetAttribute(plan->desc,
          HIPBLASLT_MATMUL_DESC_TRANSB,&transB,sizeof(transB)),"Set transB"))return nullptr;
    if(check_lt(hipblasLtMatrixLayoutCreate(&plan->A,HIP_R_16BF,K,M,K),
          "Matrix A layout"))return nullptr;
    if(check_lt(hipblasLtMatrixLayoutCreate(&plan->B,HIP_R_16BF,K,N,K),
          "Matrix B layout"))return nullptr;
    if(check_lt(hipblasLtMatrixLayoutCreate(&plan->C,HIP_R_16BF,M,N,M),
          "Matrix output layout"))return nullptr;
    hipblasLtMatmulPreference_t preference{};
    if(check_lt(hipblasLtMatmulPreferenceCreate(&preference),
          "PreferenceCreate"))return nullptr;
    size_t maximum_workspace=32ULL*1024*1024;
    if(check_lt(hipblasLtMatmulPreferenceSetAttribute(preference,
          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &maximum_workspace,sizeof(maximum_workspace)),"Max Workspace")) {
        hipblasLtMatmulPreferenceDestroy(preference);
        return nullptr;
    }
    hipblasLtMatmulHeuristicResult_t algorithms[32]{};
    int found_algorithms=0;
    const int heur=check_lt(hipblasLtMatmulAlgoGetHeuristic(context.handle,
          plan->desc,plan->A,plan->B,plan->C,plan->C,
          preference,32,algorithms,&found_algorithms),"BF16 GEMM heuristic");
    hipblasLtMatmulPreferenceDestroy(preference);
    if(heur)return nullptr;
    int chosen=-1;
    for(int i=0;i<found_algorithms;i++){
        if(algorithms[i].state==HIPBLAS_STATUS_SUCCESS
           && algorithms[i].workspaceSize<=maximum_workspace){
            if(algorithms[i].workspaceSize==0){chosen=i;break;}
            if(chosen<0)chosen=i;
        }
    }
    if(chosen<0){ERROR="No native gfx1103 BF16 hipBLASLt kernel";return nullptr;}
    plan->algo=algorithms[chosen].algo;
    plan->workspace_size=algorithms[chosen].workspaceSize;
    if(plan->workspace_size){
        if(check_hip(hipMalloc(&plan->workspace,plan->workspace_size),
           "hipMalloc hipBLASLt workspace"))return nullptr;
    }
    auto*raw=plan.get();
    context.plans.emplace(key,std::move(plan));
    return raw;
}
} // namespace

extern "C" {

const char* tv_hip_last_error() {
    return ERROR.c_str();
}
void* tv_hip_new() {
    ERROR.clear();
    if(!compatible_device())return nullptr;
    auto*ctx=new(std::nothrow)Context;
    if(!ctx){ERROR="Out of memory for native hipBLASLt context";return nullptr;}
    if(check_lt(hipblasLtCreate(&ctx->handle),"hipblasLtCreate")){
        delete ctx;return nullptr;
    }
    return ctx;
}
void tv_hip_destroy(void*handle) {
    delete static_cast<Context*>(handle);
}
void* tv_hip_load_weight(void*handle,const uint8_t*bytes,size_t byte_length) {
    ERROR.clear();
    if(!handle||!bytes||!byte_length||byte_length%2) {
        ERROR="Invalid BF16 weight payload";return nullptr;
    }
    auto& ctx=*static_cast<Context*>(handle);
    std::lock_guard<std::mutex> lock(ctx.mutex);
    void*device=nullptr;
    if(check_hip(hipMalloc(&device,byte_length),"hipMalloc exact BF16 weight"))return nullptr;
    if(check_hip(hipMemcpy(device,bytes,byte_length,hipMemcpyHostToDevice),
                 "hipMemcpy BF16 weight")) {
        hipFree(device);return nullptr;
    }
    return device;
}
int tv_hip_free_weight(void*handle,void*weight) {
    if(!handle||!weight)return fail("invalid HIP weight handle",-1);
    auto& ctx=*static_cast<Context*>(handle);
    std::lock_guard<std::mutex> lock(ctx.mutex);
    return check_hip(hipFree(weight),"hipFree BF16 weight");
}
int tv_hip_read_weight(
    void*handle,const void*weight,size_t offset,uint8_t*out,size_t size
) {
    if(!handle||!weight||!out||!size)return fail("invalid hip readback",-1);
    auto& ctx=*static_cast<Context*>(handle);
    std::lock_guard<std::mutex> lock(ctx.mutex);
    return check_hip(hipMemcpy(out,((const char*)weight)+offset,size,
        hipMemcpyDeviceToHost),"hipMemcpy weight readback");
}
int tv_hip_matrix_multiply(
    void*handle,const void*weight,const uint8_t*input,uint8_t*output,
    int inner,int rows,int tokens
) {
    ERROR.clear();
    if(!handle||!weight||!input||!output
      ||inner<=0||rows<=0||tokens<=0||tokens>512
      ||inner>32768||rows>500000) {
        return fail("invalid BF16 matrix dimensions/arguments",-1);
    }
    auto& ctx=*static_cast<Context*>(handle);
    std::lock_guard<std::mutex> lock(ctx.mutex);
    const size_t insize=(size_t)inner*tokens*2;
    const size_t outsize=(size_t)rows*tokens*2;
    if(!grow(&ctx.input,&ctx.input_size,insize,"hipMalloc BF16 input"))return 400;
    if(!grow(&ctx.output,&ctx.output_size,outsize,"hipMalloc BF16 output"))return 401;
    if(check_hip(hipMemcpy(ctx.input,input,insize,hipMemcpyHostToDevice),
          "BF16 matrix input H2D"))return 402;
    auto*plan=make_plan(ctx,rows,tokens,inner);
    if(!plan)return 403;
    const float alpha=1.0f,beta=0.0f;
    if(check_lt(hipblasLtMatmul(ctx.handle,plan->desc,&alpha,
          weight,plan->A,ctx.input,plan->B,&beta,
          ctx.output,plan->C,ctx.output,plan->C,
          &plan->algo,plan->workspace,plan->workspace_size,0),
          "native gfx1103 hipBLASLt BF16 matmul"))return 404;
    if(check_hip(hipMemcpy(output,ctx.output,outsize,hipMemcpyDeviceToHost),
         "BF16 matrix output D2H"))return 405;
    return 0;
}
}

