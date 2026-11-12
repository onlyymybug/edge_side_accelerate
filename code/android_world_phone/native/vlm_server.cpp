#include "geniex.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static int64_t nanos(Clock::time_point t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}
struct RequestSpan { std::string id; int64_t start,end,compute_start,compute_end; };
static double seconds(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now()-t).count();
}
static std::string quote(const std::string& s) {
  std::ostringstream o; o << '"';
  for (unsigned char c : s) {
    if(c=='"' || c=='\\') o << '\\' << c;
    else if(c=='\n') o << "\\n";
    else if(c=='\r') o << "\\r";
    else if(c=='\t') o << "\\t";
    else if(c<32) { const char* h="0123456789abcdef"; o << "\\u00" << h[c>>4] << h[c&15]; }
    else o << c;
  }
  o << '"'; return o.str();
}
static void emit(const std::string& s) { std::cout << "@@AWJSON@@" << s << std::endl; }
static void check(int rc, const char* operation) {
  if(rc<0) throw std::runtime_error(std::string(operation)+": "+std::to_string(rc));
}
struct StreamTime { Clock::time_point start; double first=-1; };
static bool token(const char*, void* p) {
  auto* t=static_cast<StreamTime*>(p); if(t->first<0) t->first=seconds(t->start); return true;
}
int main(int argc,char** argv) {
  if(argc!=2) { std::cerr << "usage: vlm_server MODEL_ANCHOR\n"; return 2; }
  geniex_VLM* vlm=nullptr;
  try {
    auto start=Clock::now();
    check(geniex_init(),"init");
    geniex_set_log_level(GENIEX_LOG_LEVEL_ERROR);
    geniex_VlmCreateInput create{};
    create.model_path=argv[1]; create.plugin_id="qairt"; create.device_id="NPU";
    create.config.power_mode=GENIEX_POWER_MODE_BURST;
    check(geniex_vlm_create(&create,&vlm),"create");
    auto ready=Clock::now();
    emit("{\"event\":\"ready\",\"load_seconds\":"+std::to_string(seconds(start))+
         ",\"ready_steady_ns\":"+std::to_string(nanos(ready))+"}");
    auto previous_end=ready;
    std::vector<RequestSpan> spans;
    std::string line;
    while(std::getline(std::cin,line)) {
      if(line=="shutdown") break;
      auto accepted=Clock::now();
      int64_t compute_start=0,compute_end=0;
      std::vector<std::string> parts; std::stringstream in(line); std::string p;
      while(std::getline(in,p,'\t')) parts.push_back(p);
      std::string id=parts.empty()?"":parts[0];
      geniex_VlmApplyChatTemplateOutput templ{};
      geniex_VlmGenerateOutput out{};
      try {
        if(parts.size()<4) throw std::runtime_error("expected id, prompt path, one or more image paths, max tokens");
        std::ifstream file(parts[1]);
        if(!file) throw std::runtime_error("cannot open prompt");
        std::string prompt((std::istreambuf_iterator<char>(file)),{});
        std::vector<geniex_Path> images;
        for(size_t i=2;i+1<parts.size();++i) images.push_back(parts[i].c_str());
        auto begin=Clock::now();
        compute_start=nanos(begin);
        check(geniex_vlm_reset(vlm),"reset");
        std::vector<geniex_VlmContent> contents={{"text",prompt.c_str()}};
        for(auto image:images) contents.push_back({"image",image});
        geniex_VlmChatMessage msg{}; msg.role="user"; msg.contents=contents.data(); msg.content_count=contents.size();
        geniex_VlmApplyChatTemplateInput ti{}; ti.messages=&msg; ti.message_count=1;
        ti.parallel_tool_calls=true;
        check(geniex_vlm_apply_chat_template(vlm,&ti,&templ),"template");
        geniex_SamplerConfig sampler{}; sampler.temperature=0; sampler.seed=42; sampler.repetition_penalty=1;
        geniex_GenerationConfig config{}; config.max_tokens=std::stoi(parts.back());
        config.sampler_config=&sampler; config.image_paths=images.data(); config.image_count=images.size();
        geniex_VlmGenerateInput gi{}; gi.prompt_utf8=templ.formatted_text; gi.config=&config;
        StreamTime st{Clock::now()}; gi.on_token=token; gi.user_data=&st;
        check(geniex_vlm_generate(vlm,&gi,&out),"generate");
        auto generated=Clock::now();
        compute_end=nanos(generated);
        auto& v=out.profile_data;
        std::ostringstream result;
        result << "{\"event\":\"response\",\"id\":" << quote(id)
          << ",\"image_count\":" << images.size()
          << ",\"text\":" << quote(out.full_text?out.full_text:"")
          << ",\"elapsed_seconds\":" << (compute_end-compute_start)/1e9
          << ",\"request_accepted_steady_ns\":" << nanos(accepted)
          << ",\"compute_start_steady_ns\":" << compute_start
          << ",\"compute_end_steady_ns\":" << compute_end
          << ",\"service_idle_before_request_seconds\":" << (nanos(accepted)-nanos(previous_end))/1e9
          << ",\"callback_ttft_seconds\":" << st.first
          << ",\"ttft_us\":" << v.ttft << ",\"media_time_us\":" << v.media_time
          << ",\"prompt_time_us\":" << v.prompt_time << ",\"decode_time_us\":" << v.decode_time
          << ",\"prompt_tokens\":" << v.prompt_tokens << ",\"generated_tokens\":" << v.generated_tokens
          << ",\"decoding_speed\":" << v.decoding_speed
          << ",\"stop_reason\":" << quote(v.stop_reason?v.stop_reason:"") << "}";
        emit(result.str());
      } catch(const std::exception& e) {
        if(compute_start && !compute_end) compute_end=nanos(Clock::now());
        emit("{\"event\":\"error\",\"id\":"+quote(id)+",\"error\":"+quote(e.what())+"}");
      }
      if(templ.formatted_text) geniex_free(templ.formatted_text);
      if(out.full_text) geniex_free(out.full_text);
      previous_end=Clock::now();
      spans.push_back({id,nanos(accepted),nanos(previous_end),compute_start,compute_end});
      // QAIRT emits no tool grammar for these plain user messages.
    }
    auto session_end=Clock::now();
    int64_t handler_ns=0,compute_ns=0;
    std::ostringstream rows;
    for(size_t i=0;i<spans.size();++i) {
      auto& s=spans[i]; handler_ns+=s.end-s.start;
      if(s.compute_start) compute_ns+=s.compute_end-s.compute_start;
      if(i) rows << ',';
      rows << "{\"id\":" << quote(s.id) << ",\"accepted_ns\":" << s.start
           << ",\"handler_end_ns\":" << s.end << ",\"compute_start_ns\":" << s.compute_start
           << ",\"compute_end_ns\":" << s.compute_end << '}';
    }
    std::ostringstream stats;
    stats << "{\"event\":\"session_end\",\"clock\":\"device steady_clock nanoseconds\""
          << ",\"ready_ns\":" << nanos(ready) << ",\"shutdown_received_ns\":" << nanos(session_end)
          << ",\"ready_to_shutdown_seconds\":" << (nanos(session_end)-nanos(ready))/1e9
          << ",\"handler_busy_seconds\":" << handler_ns/1e9
          << ",\"compute_busy_seconds\":" << compute_ns/1e9
          << ",\"requests\":[" << rows.str() << "]}";
    emit(stats.str());
    geniex_vlm_destroy(vlm); geniex_deinit(); return 0;
  } catch(const std::exception& e) {
    emit("{\"event\":\"fatal\",\"error\":"+quote(e.what())+"}");
    if(vlm) geniex_vlm_destroy(vlm); geniex_deinit(); return 1;
  }
}
