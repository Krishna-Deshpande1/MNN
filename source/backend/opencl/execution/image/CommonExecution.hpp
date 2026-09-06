//
//  CommonExecution.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CommonExecution_hpp
#define CommonExecution_hpp
#include "core/Execution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#ifdef __ANDROID__
#include <android/log.h>
#endif
namespace MNN {
namespace OpenCL {

// TEMPORARY DEBUG: identify exactly which of the ~13+ OPENCL_CHECK_KERNEL(_CTOR)
// guard sites is firing when a kernel comes back null, since the two known
// buildKernel() failure paths (compile error in buildProgram(), kernel-name
// lookup failure in buildKernelWithCache()) were both instrumented and
// neither fired - meaning the null is coming from somewhere else, or one of
// these guards is catching a kernel that failed for a still-unidentified
// reason. Uses __android_log_print directly (not MNN_ERROR) because this
// build's MNN_USE_LOGCAT=false makes MNN_ERROR invisible in `adb logcat`.
// Remove both this logging and the include above after investigation.
#ifdef __ANDROID__
#define OPENCL_CHECK_KERNEL_GUARD_LOG(kernel) \
    __android_log_print(ANDROID_LOG_ERROR, "MNN_OPENCL_KERNEL_GUARD", \
                         "%s is null at %s:%d", #kernel, __FILE__, __LINE__);
#else
#define OPENCL_CHECK_KERNEL_GUARD_LOG(kernel)
#endif

// Check kernel after buildKernel in constructor; set mValid=false on failure
#define OPENCL_CHECK_KERNEL_CTOR(kernel)  \
    if (kernel == nullptr) {              \
        OPENCL_CHECK_KERNEL_GUARD_LOG(kernel) \
        mValid = false;                   \
        return;                           \
    }

// Check kernel after buildKernel in onResize/onEncode; return NOT_SUPPORT on failure
#define OPENCL_CHECK_KERNEL(kernel)       \
    if (kernel == nullptr) {              \
        OPENCL_CHECK_KERNEL_GUARD_LOG(kernel) \
        return NOT_SUPPORT;               \
    }

// Wrap 'new Execution' in creator: validate then return (or nullptr on failure)
inline Execution* checkExeValid(Execution* exe) {
    if (exe != nullptr && !exe->valid()) {
        delete exe;
        return nullptr;
    }
    return exe;
}
#define OPENCL_CREATOR_CHECK(p) return checkExeValid(p)

// Check onAcquireBuffer in constructor; set mValid=false on failure
#define OPENCL_CHECK_ALLOC_CTOR(expr)     \
    if (!(expr)) {                        \
        mValid = false;                   \
        return;                           \
    }

// Check onAcquireBuffer in onResize/onEncode; return OUT_OF_MEMORY on failure
#define OPENCL_CHECK_ALLOC(expr)          \
    if (!(expr)) {                        \
        return OUT_OF_MEMORY;             \
    }

// Check pointer (e.g. ConvolutionCommon::load) in constructor; set mValid=false on failure
#define OPENCL_CHECK_PTR_CTOR(ptr)        \
    if (ptr == nullptr) {                 \
        mValid = false;                   \
        return;                           \
    }

struct Unit {
    std::shared_ptr<KernelWrap> kernel;
    cl::NDRange globalWorkSize;
    cl::NDRange localWorkSize;
};

class CommonExecution : public Execution {
public:
    CommonExecution(Backend *backend, const MNN::Op *Op);
    virtual ~CommonExecution(){
        if(mRecording != NULL){
#ifdef MNN_USE_LIB_WRAPPER
            clReleaseRecordingQCOM(mRecording);
#endif
        }
    }
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
        return NO_ERROR;
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    std::vector<Unit> mUnits;
    const MNN::Op *mOp;
    OpType mOpType;
    cl_recording_qcom mRecording{NULL};
    std::vector<RecordUpdateInfo*> mOpRecordUpdateInfo;
};
} // namespace OpenCL
} // namespace MNN
#endif /* CommonExecution_hpp */
