// Optional LD_PRELOAD instrumentation. All measured intervals use the phone clock.
#include "geniex.h"
#include <dlfcn.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <mutex>
#include <algorithm>

using Clock=std::chrono::steady_clock;
static double now(){return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();}
static void* resolve(const char* symbol,bool api=false){
  const char* path=getenv(api?"AW_DIAG_API_PATH":"AW_DIAG_CORE_PATH");
  void* h=path?dlopen(path,RTLD_NOW|RTLD_NOLOAD):nullptr;
  void* p=h?dlsym(h,symbol):dlsym(RTLD_NEXT,symbol);
  if(!p){fprintf(stderr,"AWDIAG resolve failed: %s %s\n",symbol,dlerror());abort();}
  return p;
}
struct Event {size_t phase,shard,cl;double seconds;std::string name;};
static std::mutex mu;
static std::vector<Event> graphs,shards,kvs;
static std::vector<double> tokens;
static thread_local size_t phase_now=99,shard_now=99,cl_now=0;
static unsigned request_index=0;
static void emit_events(const char* key,const std::vector<Event>& events){
  printf(",\"%s\":[",key);
  for(size_t i=0;i<events.size();i++){
    const auto&e=events[i];
    printf("%s{\"phase\":%zu,\"shard\":%zu,\"cl\":%zu,\"seconds\":%.9f,\"graph\":\"%s\"}",i?",":"",e.phase,e.shard,e.cl,e.seconds,e.name.c_str());
  }printf("]");
}
extern "C" bool diag_promote(void*,size_t,size_t,size_t) asm("_ZN6geniex8LLMModel9promoteCLEmmm");
extern "C" bool diag_promote(void* self,size_t required,size_t capacity,size_t stride){
  static auto orig=(bool(*)(void*,size_t,size_t,size_t))resolve("_ZN6geniex8LLMModel9promoteCLEmmm");
  const char* force=getenv("AW_FORCE_CL");size_t cl=force?strtoull(force,nullptr,10):0;
  if(cl)required=std::max(required,cl-(capacity==0?128:1));
  return orig(self,required,capacity,stride);
}
extern "C" void diag_shard(void*,size_t,size_t,size_t,const void*,const void*) asm("_ZN6geniex8LLMModel8runShardEmmmRKNS_13LLMRunContextERKNSt6__ndk18functionIFvRNS_5GraphEEEE");
extern "C" void diag_shard(void*self,size_t shard,size_t phase,size_t idx,const void*ctx,const void*fn){
  static auto orig=(void(*)(void*,size_t,size_t,size_t,const void*,const void*))resolve("_ZN6geniex8LLMModel8runShardEmmmRKNS_13LLMRunContextERKNSt6__ndk18functionIFvRNS_5GraphEEEE");
  static auto width=(size_t(*)(void*,size_t,size_t))resolve("_ZNK6geniex8LLMModel11kvMaskWidthEmm");
  phase_now=phase;shard_now=shard;cl_now=width(self,phase,idx);
  double t=now();orig(self,shard,phase,idx,ctx,fn);double dt=now()-t;
  {std::lock_guard<std::mutex> lock(mu);shards.push_back({phase,shard,cl_now,dt,{}});}
  phase_now=99;shard_now=99;cl_now=0;
}
// Graph::execute is inlined for LLM calls in this build. Interpose the QNN
// wrapper instead. GraphInfo's ABI begins with graph handle, then char* name;
// verified against this library's disassembly (graphName loaded from +8).
struct GraphInfoPrefix {void* handle;const char* name;};
extern "C" bool diag_qnn(void*,GraphInfoPrefix*,const void*,void*,void*) asm("_ZN6QnnApi12graphExecuteEPN15qnn_wrapper_api9GraphInfoEPK12Qnn_Tensor_tPS3_RNSt6__ndk13mapINS7_12basic_stringIcNS7_11char_traitsIcEENS7_9allocatorIcEEEENS7_4pairIdtEENS7_4lessISE_EENSC_INSF_IKSE_SG_EEEEEE");
extern "C" bool diag_qnn(void*self,GraphInfoPrefix*info,const void*input,void*output,void*times){
  static auto orig=(bool(*)(void*,GraphInfoPrefix*,const void*,void*,void*))resolve("_ZN6QnnApi12graphExecuteEPN15qnn_wrapper_api9GraphInfoEPK12Qnn_Tensor_tPS3_RNSt6__ndk13mapINS7_12basic_stringIcNS7_11char_traitsIcEENS7_9allocatorIcEEEENS7_4pairIdtEENS7_4lessISE_EENSC_INSF_IKSE_SG_EEEEEE");
  double t=now();bool ok=orig(self,info,input,output,times);double dt=now()-t;
  {std::lock_guard<std::mutex> lock(mu);graphs.push_back({phase_now,shard_now,cl_now,dt,info->name});}return ok;
}
extern "C" void diag_kv(void*,size_t,size_t,size_t,size_t) asm("_ZN6geniex8LLMModel8updateKVEmmmm");
extern "C" void diag_kv(void*self,size_t shard,size_t phase,size_t dst,size_t n){
  static auto orig=(void(*)(void*,size_t,size_t,size_t,size_t))resolve("_ZN6geniex8LLMModel8updateKVEmmmm");
  double t=now();orig(self,shard,phase,dst,n);double dt=now()-t;
  std::lock_guard<std::mutex> lock(mu);kvs.push_back({phase,shard,0,dt,{}});
}
struct Callback {geniex_token_callback original;void* data;double start;};
static bool token(const char*text,void*opaque){auto*c=(Callback*)opaque;
  {std::lock_guard<std::mutex> lock(mu);tokens.push_back(now()-c->start);}
  return c->original?c->original(text,c->data):true;
}
extern "C" int32_t geniex_vlm_generate(geniex_VLM*h,const geniex_VlmGenerateInput*in,geniex_VlmGenerateOutput*out){
  static auto orig=(decltype(&geniex_vlm_generate))resolve("geniex_vlm_generate",true);
  {std::lock_guard<std::mutex> lock(mu);graphs.clear();shards.clear();kvs.clear();tokens.clear();}
  Callback cb{in->on_token,in->user_data,now()};auto input=*in;input.on_token=token;input.user_data=&cb;
  int32_t ret=orig(h,&input,out);double elapsed=now()-cb.start;
  std::lock_guard<std::mutex> lock(mu);
  printf("@@AWDIAG@@{\"request_index\":%u,\"return_code\":%d,\"generate_seconds\":%.9f,\"token_seconds\":[",request_index++,ret,elapsed);
  for(size_t i=0;i<tokens.size();i++)printf("%s%.9f",i?",":"",tokens[i]);printf("]");
  emit_events("graphs",graphs);emit_events("shards",shards);emit_events("kv_updates",kvs);printf("}\n");fflush(stdout);return ret;
}
