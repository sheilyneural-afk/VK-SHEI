#pragma once

// Parse-only declarations for Clang AST inventory generation. They deliberately
// describe only the names and shapes used by native/hipblaslt_bridge.cpp; they
// are never compiled or linked into M-SLIM.
using hipError_t = int;
inline constexpr hipError_t hipSuccess = 0;
struct hipDeviceProp_t {
    char gcnArchName[256];
};
inline constexpr int hipMemcpyHostToDevice = 1;
inline constexpr int hipMemcpyDeviceToHost = 2;
const char* hipGetErrorString(hipError_t);
hipError_t hipGetDeviceCount(int*);
hipError_t hipGetDeviceProperties(hipDeviceProp_t*, int);
hipError_t hipFree(void*);
hipError_t hipMalloc(void**, unsigned long);
hipError_t hipMemcpy(void*, const void*, unsigned long, int);

using hipblasStatus_t = int;
using hipblasLtHandle_t = void*;
using hipblasLtMatmulDesc_t = void*;
using hipblasLtMatrixLayout_t = void*;
using hipblasLtMatmulPreference_t = void*;
using hipblasLtMatmulAlgo_t = void*;
using hipblasOperation_t = int;
inline constexpr hipblasStatus_t HIPBLAS_STATUS_SUCCESS = 0;
inline constexpr int HIPBLAS_COMPUTE_32F = 0;
inline constexpr int HIP_R_32F = 0;
inline constexpr int HIP_R_16BF = 1;
inline constexpr hipblasOperation_t HIPBLAS_OP_T = 1;
inline constexpr hipblasOperation_t HIPBLAS_OP_N = 0;
inline constexpr int HIPBLASLT_MATMUL_DESC_TRANSA = 1;
inline constexpr int HIPBLASLT_MATMUL_DESC_TRANSB = 2;
inline constexpr int HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 3;
struct hipblasLtMatmulHeuristicResult_t {
    hipblasStatus_t state;
    unsigned long workspaceSize;
    hipblasLtMatmulAlgo_t algo;
};
hipblasStatus_t hipblasLtCreate(...);
hipblasStatus_t hipblasLtDestroy(...);
hipblasStatus_t hipblasLtMatmulDescCreate(...);
hipblasStatus_t hipblasLtMatmulDescDestroy(...);
hipblasStatus_t hipblasLtMatmulDescSetAttribute(...);
hipblasStatus_t hipblasLtMatrixLayoutCreate(...);
hipblasStatus_t hipblasLtMatrixLayoutDestroy(...);
hipblasStatus_t hipblasLtMatmulPreferenceCreate(...);
hipblasStatus_t hipblasLtMatmulPreferenceDestroy(...);
hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(...);
hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(...);
hipblasStatus_t hipblasLtMatmul(...);
