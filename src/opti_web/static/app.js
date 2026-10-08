"use strict";
const $ = (id) => document.getElementById(id);
const state = {job:null, stream:null, busy:false, finalized:false, lastSeq:0, stageCount:0, seen:new Set(), assistant:null};
const baseCommand = "A 선반의 빨간 용기를 출고대에 가져다 놓아줘.";
const labels = {
  redbin01:"빨간 소형 용기", station_A:"A 입고대", station_B:"B 조립대", slot_B_1:"첫 번째 자리", tray_slot_1:"트레이 첫 번째 자리",
  bluebin01:"파란 소형 용기",greenbin01:"초록 소형 용기",station_home:"로봇 시작 위치",station_rack_A:"A 선반",station_rack_B:"B 선반",station_rack_C:"C 선반",station_out:"출고대",slot_out_1:"첫 번째 자리",
  corridor_lower:"아래 통로", corridor_upper:"위 통로",
  navigate:"작업 장소로 이동", observe:"물체 위치 확인", pick:"물체 집기", place_on_tray:"트레이에 적재", stow_arms:"운반 자세 유지", pick_from_tray:"트레이에서 꺼내기", place:"목적지에 놓기", verify_place:"놓기 결과 확인",
  base_path_tracking:"주행 경로 추종",base_effort:"주행 명령 크기",base_command_change:"주행 명령 부드러움",arm_tcp_tracking:"손끝 목표 추종",arm_joint_velocity:"팔 관절 명령 크기",arm_command_change:"팔 명령 부드러움",
  base_speed_max:"주행 명령 속도 상한",base_accel_max:"주행 명령 변화율 상한",base_yaw_rate_max:"주행 명령 회전속도 상한",obstacle_clearance_min:"장애물 여유 하한",arm_joint_speed_max:"팔 관절 명령 속도 상한",arm_joint_accel_max:"팔 관절 명령 변화율 상한",
  starting:"실행 준비",cancelling:"실행 중지 요청",llm:"LLM 지시 해석",planning:"LLM 작업 계획",interpret:"지시 해석",spec:"작업 명세 검증",constraints:"최적화 조건 구성",validation:"명세 검증",path:"이동 경로 계산",nav2:"이동 경로 계산",route:"이동 경로 계산",simulation:"로봇 실행 준비",physics:"로봇 물리 실행",drive:"주행",approach_source:"입고대에 접근",tray:"트레이 적재",unload:"트레이에서 꺼내기",verify:"결과 검증",verification:"결과 검증",complete:"최종 검증",runtime:"로봇 작업 실행",settle:"초기 상태 안정화",approach:"집기 위치 접근",close:"그리퍼 닫기",lift:"물체 들어 올리기",hold:"파지 유지",transfer:"물체 옮기기",lower:"놓기 위치로 내리기",open:"그리퍼 열기",retreat:"팔 물러나기",place_settle:"놓은 물체 안정화",
};
Object.assign(labels, {simulation_start:"시뮬레이션 초기화",navigation_plan:"Nav2 경로 계획",undock:"선반에서 후진",dock:"목표 구역에 도킹",approach_settle:"접근 자세 안정화",transfer_settle:"트레이 위 자세 안정화",lower_settle:"트레이 적재 자세 확인"});
Object.assign(labels,{dynamic_people_active:"보행자 이동·LiDAR 관측",dynamic_person_setup:"보행자 시나리오 준비",dynamic_person_walk:"작업자 횡단 시작",dynamic_obstacle_setup:"카트 차단물 준비",dynamic_cart_release:"외부 카트 이동 시작",dynamic_braking:"장애물 감속·정지 요청",dynamic_waiting:"실제 정지 확인·통로 대기",dynamic_resumed:"통로 확인 후 재출발 명령",dynamic_emergency_stop:"오류 후 실제 정지 확인"});
for (const [name, title] of Object.entries({hover:"트레이 위 접근",hover_settle:"트레이 관측 자세 확인",approach:"트레이 물체로 접근",approach_settle:"재파지 자세 확인",close:"트레이 물체 재파지",lift:"트레이에서 들어 올리기",hold:"재파지 유지 확인",transfer:"출고대로 물체 옮기기",transfer_settle:"출고대 배치 준비",lower:"출고대에 내려놓기",lower_settle:"출고대 지지 자세 확인",open:"출고대에서 그리퍼 해제",retreat:"출고대에서 팔 물러나기",place_settle:"최종 배치 안정화"})) labels['unload_'+name]=title;
const groupFor = (stage) => {
  stage=String(stage||"");
  if(stage === 'navigation_plan')return 2;
  if(stage.startsWith('dynamic_'))return 3;
  if(stage.startsWith('unload_') || ['simulation_start','undock','dock','approach_settle','transfer_settle','lower_settle'].includes(stage))return 3;
  if (["llm","planning","interpret","recognize"].includes(stage)) return 0;
  if (["spec","constraints","validation","validate"].includes(stage)) return 1;
  if (["path","nav2","route","path_planning"].includes(stage)) return 2;
  if (["verify","verification","complete","completed","result","verify_place"].includes(stage)) return 4;
  if (["simulation","physics","runtime","drive","approach_source","navigate","tray","unload","settle","approach","close","lift","hold","transfer","lower","open","retreat","place_settle","pick","place","place_on_tray","pick_from_tray","stow_arms","observe"].includes(stage)) return 3;
  return null;
};
const label = (key) => labels[key] || String(key || "—");
const element = (tag, cls, text) => {const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=String(text);return node;};
const number = (value) => typeof value==="number"&&Number.isFinite(value)?value.toLocaleString("ko-KR",{maximumSignificantDigits:12}):String(value ?? "—");
const readable = (value) => typeof value==="string"?value:value==null?"":JSON.stringify(value,null,2);
// Strip a machine reason-code prefix only in visible Korean feedback. Raw
// event/spec/result JSON remains unchanged for diagnosis and provenance.
const humanMessage = (value) => readable(value).replace(/^[a-z][a-z0-9_]*:\s*(?=[가-힣])/, "");
const timeLabel = (timestamp) => {const date=new Date(timestamp);return Number.isNaN(date.valueOf())?"":date.toLocaleTimeString("ko-KR",{hour12:false,hour:"2-digit",minute:"2-digit",second:"2-digit"});};

function message(text,role="assistant",kind="") {
  const row=element("article",`chat-message ${role} ${kind}`);
  if(role!=="user")row.append(element("span","avatar","F"));
  const content=element("div","message-content");content.append(element("span","message-name",role==="user"?"내 작업 지시":"Factory Copilot"));
  const bubble=element("div","message-bubble",text);content.append(bubble);row.append(content);$("messages").append(row);scrollChat();return bubble;
}
function scrollChat(){ $("messages").scrollTop=$("messages").scrollHeight; }
function updatePeopleCountLock(){ $("people-count").disabled=state.busy||!$("moving-person").checked; }
function setBusy(value){state.busy=value;$("send").disabled=value;$("command").disabled=value;$("show-sim").disabled=value;$("moving-person").disabled=value;updatePeopleCountLock();$("cancel").hidden=!value;document.querySelectorAll("[data-example]").forEach(b=>b.disabled=value);}
function updatePath(group){document.querySelectorAll("#stage-path>div").forEach(node=>{const i=Number(node.dataset.group);node.classList.toggle("seen",state.seen.has(i));node.classList.toggle("active",group===i);});}
function stage(event){
  const group=groupFor(event.stage);if(group!==null)state.seen.add(group);updatePath(group);
  const title=label(event.stage);const detail=humanMessage(event.detail);
  $("stage-title").textContent=title;$("stage-detail").textContent=detail||"진행 메시지를 받았습니다.";
  state.stageCount++;$("event-count").textContent=state.stageCount;$("activity-empty").hidden=true;
  const row=element("li");const heading=element("div","activity-title",title);heading.append(element("span","activity-time",timeLabel(event.received_at)));row.append(heading);if(detail)row.append(element("p","activity-detail",detail));$("activity").append(row);$("activity").scrollTop=$("activity").scrollHeight;
  if(state.assistant)state.assistant.textContent=`${title}\n${detail||"실행 단계의 진행 메시지를 받고 있습니다."}`;
}
function numericRows(target,items,key,value){
  target.replaceChildren();if(!Array.isArray(items))return;
  for(const item of items){if(!item||typeof item!=="object")continue;const row=element("div","numeric-row");row.append(element("span",null,label(item[key])));row.append(element("span","numeric-value",`${number(item[value])}${item.unit?" "+item.unit:""}`));target.append(row);}
}
function renderSpec(event){
  const spec=event.spec?.task||event.spec;if(!spec||typeof spec!=="object"||event.validated===false)return;
  $("plan-empty").hidden=true;$("plan-content").hidden=false;$("plan-state").textContent="명세 수신";$("plan-state").classList.remove("neutral");
  const summary=$("entity-summary");summary.replaceChildren();
  const slot=spec.destination_slot_id||(spec.ordered_skills||[]).slice().reverse().find(s=>s.skill_id==="place")?.slot_id;
  for(const [name,value] of [["물건",spec.object_id],["출발",spec.source_id],["목적지",spec.destination_id]]){const row=element("div","entity-row");row.append(element("span","entity-label",name));row.append(element("span","entity-value",label(value)+(name==="목적지"&&slot?` · ${label(slot)}`:"")));summary.append(row);}
  $("skills").replaceChildren();(spec.ordered_skills||[]).forEach((skill,i)=>{const row=element("li");row.append(element("span","skill-number",i+1),element("span",null,label(skill.skill_id)));const destination=skill.station_id||skill.slot_id;if(destination)row.append(element("span","skill-target",label(destination)));$("skills").append(row);});
  $("zones").replaceChildren();const zones=spec.forbidden_zone_ids;if(Array.isArray(zones)&&zones.length)zones.forEach(zone=>$("zones").append(element("span","zone-chip",`${label(zone)} 금지`)));else $("zones").append(element("span",null,Array.isArray(zones)?"지정된 금지 구역 없음":"금지 구역 정보 없음"));
  numericRows($("bounds"),spec.constraints,"constraint_id","bound");numericRows($("weights"),spec.objective_terms,"term_id","weight");
  $("formulas").replaceChildren();
  for(const formula of Array.isArray(event.formulas)?event.formulas:[]){
    if(typeof formula?.latex!=="string"||formula.latex.length>8192)continue;
    const card=element("div","formula");card.append(element("h4",null,formula.title||"적용한 조건"));const math=element("div","formula-content");card.append(math);$("formulas").append(card);
    if(window.katex){try{window.katex.render(formula.latex,math,{displayMode:true,throwOnError:true,trust:false,strict:"warn"});}catch(_){math.append(element("p","math-error","이 조건의 수식 표시를 처리하지 못했습니다. 명세 값은 위 목록에서 확인할 수 있습니다."));}}else math.append(element("p","math-error","수식 렌더러를 불러오지 못했습니다. 명세 값은 위 목록에서 확인할 수 있습니다."));
  }
  const assumptions=$("assumptions");assumptions.replaceChildren();
  const notes=event.assumptions||event.limitations;
  if(Array.isArray(notes))notes.forEach(note=>assumptions.append(element("p",null,readable(note))));
  else {assumptions.append(element("p",null,"물리 시뮬레이션에서 실행하는 작업입니다."));assumptions.append(element("p",null,"목적함수·명령 한계는 검증된 명세 값이며, 로봇의 하드웨어 정격과 구분됩니다."));assumptions.append(element("p",null,"실행 순서는 지원된 작업 기술로 구성됩니다. 수식은 전달된 최적화 조건만 표시합니다."));}
  $("raw-spec").textContent=JSON.stringify(spec,null,2);
}
function artifactUrl(path){
  if(typeof path!=="string"||!state.job)return null;
  const prefix=state.job.output_directory+"/";
  if(path.startsWith(prefix))path=path.slice(prefix.length);
  if(path.startsWith("/")||path.includes(":")||path.split("/").includes(".."))return null;
  return `/artifacts/${state.job.id}/${path.split("/").map(encodeURIComponent).join("/")}`;
}
function media(artifacts){
  const values=Array.isArray(artifacts)?artifacts:Object.values(artifacts||{});const files=values.filter(p=>typeof p==="string"&&artifactUrl(p));if(!files.length)return;
  $("media-panel").hidden=false;$("media").replaceChildren();
  const video=files.find(p=>/\.(mp4|webm)$/i.test(p));const image=files.find(p=>/\.(png|jpg|jpeg|webp)$/i.test(p));
  if(video){const v=element("video");v.controls=true;v.preload="metadata";v.src=artifactUrl(video);if(image)v.poster=artifactUrl(image);$("media").append(v);}else if(image){const img=element("img");img.src=artifactUrl(image);img.alt="실제 시뮬레이션에서 저장된 로봇 화면";$("media").append(img);}
  const links=element("div","media-links");files.forEach(path=>{const a=element("a",null,path.split("/").pop());a.href=artifactUrl(path);a.target="_blank";a.rel="noopener";links.append(a);});$("media").append(links);
}
function result(event){
  if(state.finalized)return;state.finalized=true;
  setBusy(false);$("cancel").disabled=false;if(state.stream){state.stream.close();state.stream=null;}
  const accepted=event.accepted===true;$("stage-path").classList.toggle("failed",!accepted);
  $("stage-title").textContent=event.cancelled?"실행을 중지했습니다":accepted?"작업 검증을 통과했습니다":"작업이 완료되지 않았습니다";$("stage-detail").textContent=humanMessage(event.detail)||"최종 결과를 받았습니다.";
  if(!state.assistant)state.assistant=message("", "assistant",accepted?"success":"error");
  state.assistant.textContent=humanMessage(event.detail)||(accepted?"최종 검증을 통과했습니다.":"최종 검증을 통과하지 못했습니다.");state.assistant.closest("article").classList.add(accepted?"success":"error");
  const card=element("div","message-result");card.append(element("div",`result-badge ${accepted?"":"bad"}`,event.cancelled?"실행 중지":accepted?"검증 통과":"실행 거부 또는 검증 실패"));
  if(typeof event.summary==="string")card.append(element("div","result-summary",event.summary));
  const checks=event.summary?.checks||event.checks;if(checks&&typeof checks==="object"&&!Array.isArray(checks)){const entries=Object.values(checks).filter(v=>typeof v==="boolean");if(entries.length)card.append(element("div","result-summary",`검증 항목 ${entries.filter(Boolean).length} / ${entries.length} 통과`));}
  const details=element("details","result-raw");details.append(element("summary",null,"검증 결과 원문 보기"));details.append(element("pre",null,JSON.stringify(event,null,2)));card.append(details);state.assistant.append(card);media(event.artifacts||event.summary?.artifacts);scrollChat();
}
function consume(event){
  if(event.seq&&event.seq<=state.lastSeq)return;state.lastSeq=event.seq||state.lastSeq;
  if(event.type==="stage")stage(event);else if(event.type==="spec")renderSpec(event);else if(event.type==="result")result(event);else if(event.type==="preview"&&event.path)media([event.path]);else if(event.type==="message"&&event.detail)message(readable(event.detail),"assistant",event.level==="warning"?"error":"");
}
function connect(job){
  if(state.stream)state.stream.close();state.job=job;state.lastSeq=0;
  const stream=new EventSource(`/api/jobs/${job.id}/events`);state.stream=stream;
  stream.onmessage=(message)=>{if(state.stream!==stream||state.job?.id!==job.id)return;try{consume(JSON.parse(message.data));}catch(_){$("connection").textContent="진행 메시지 확인 중";}};
  stream.onopen=()=>{if(state.stream===stream)connection(true);};stream.onerror=()=>{if(state.stream===stream&&state.busy)connection(false,"진행 연결 재시도 중");};
}
function resetWork(){
  state.finalized=false;
  state.seen.clear();state.stageCount=0;state.lastSeq=0;$("stage-path").classList.remove("failed");updatePath(null);$("activity").replaceChildren();$("activity-empty").hidden=false;$("event-count").textContent="0";$("plan-empty").hidden=false;$("plan-content").hidden=true;$("plan-state").textContent="대기";$("plan-state").classList.add("neutral");$("conditions").open=false;$("media-panel").hidden=true;state.assistant=null;
}
async function api(path,body){const response=await fetch(path,body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw Error(data.error||"연결 요청을 처리하지 못했습니다.");return data;}
function connection(connected,text){const node=$("connection");node.replaceChildren(element("i"),document.createTextNode(text||(connected?"로컬 서버 연결됨":"서버 연결 확인 필요")));node.className=`connection ${connected?"connected":"offline"}`;}
async function refreshStatus(initial=false){
  try{const response=await api("/api/status");connection(true);$("availability").textContent=response.backend_ready?"한 번에 한 작업을 실행합니다. 다른 지시는 현재 작업이 끝난 뒤 보내 주세요.":"실행 백엔드가 아직 준비되지 않았습니다. 작업을 실행하려면 실행 연결이 필요합니다.";
    if(initial&&response.job){resetWork();$("show-sim").checked=response.job.show_sim===true;$("moving-person").checked=response.job.moving_person===true;$("people-count").value=response.job.people_count===1?"1":"3";message(response.job.command,"user");state.assistant=message("저장된 실행 진행을 연결하고 있습니다.");setBusy(!["completed","failed","cancelled"].includes(response.job.state));connect(response.job);}
    else if(state.busy&&response.job?.id===state.job?.id&&response.job.result){result(response.job.result);}
  }catch(_){connection(false);$("availability").textContent="로컬 서버에 연결되지 않았습니다. 서버 실행 상태를 확인해 주세요.";}
}
$("command-form").addEventListener("submit",async(event)=>{
  event.preventDefault();if(state.busy)return;const command=$("command").value.trim();if(!command)return;
  setBusy(true);message(command,"user");
  try{const job=await api("/api/jobs",{command,show_sim:$("show-sim").checked,moving_person:$("moving-person").checked,people_count:Number($("people-count").value),layout:"rack-v1"});resetWork();state.assistant=message("지시를 접수했습니다. 실제 실행 진행을 연결하고 있습니다.");connect(job);}
  catch(error){setBusy(false);message(error.message,"assistant","error");$("stage-title").textContent="지시를 실행하지 못했습니다";$("stage-detail").textContent=error.message;}
});
$("command").addEventListener("keydown",(event)=>{if(event.key==="Enter"&&!event.shiftKey&&!event.isComposing){event.preventDefault();$("command-form").requestSubmit();}});
$("moving-person").addEventListener("change",updatePeopleCountLock);
$("cancel").addEventListener("click",async()=>{if(!state.job||!state.busy)return;$("cancel").disabled=true;try{await api(`/api/jobs/${state.job.id}/cancel`,{});}catch(error){$("cancel").disabled=false;message(error.message,"assistant","error");}});
document.querySelectorAll("[data-example]").forEach(button=>button.addEventListener("click",()=>{$("command").value=button.dataset.example==="blue-slow"?"B 선반의 파란 용기를 출고대에 가져다 놓아줘. 천천히 운반해줘.":button.dataset.example==="green"?"C 선반의 초록 용기를 출고대에 가져다 놓아줘.":baseCommand;$("command").focus();}));
refreshStatus(true);setInterval(()=>refreshStatus(),4000);
