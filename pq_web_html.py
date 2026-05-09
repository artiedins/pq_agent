#!/usr/bin/env python3
# pq_web_html.py - HTML/CSS/JS template for pq_web.py
#
# Separated from pq_web.py because this file is purely UI markup and will be
# updated less frequently than the server-side logic. pq_web.py imports HTML
# from here as its single-page-app body.
#
# Code style:
# - No type hinting
# - No doc strings
# - No triple quoted multi-line strings  <-- exception: the HTML constant must use one
# - No non-ascii characters
# - Yes strategic inline comments enhancing rapid code comprehension

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PQ MINDER</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100vh;overflow:hidden;background:#111;font-family:'Courier New',Courier,monospace;font-size:14px;color:#bbb;}
.app{height:100vh;display:flex;flex-direction:column;}

/* titlebar */
.tb{background:#1c1c1c;border-bottom:2px solid #2e2e2e;padding:6px 16px;display:flex;justify-content:space-between;align-items:center;flex-shrink:0;gap:10px;}
.tname{color:#f0f0f0;letter-spacing:.1em;font-size:15px;flex-shrink:0;}
.tctrl{display:flex;gap:9px;align-items:center;flex-wrap:wrap;}
.tstat{color:#999;font-size:12px;}
.tmodels{display:flex;gap:8px;align-items:center;flex-shrink:0;}
.tmsep{color:#444;font-size:12px;}

/* model selector chips */
.mchip{display:inline-flex;align-items:center;gap:5px;border:1px solid #3a3a3a;background:#141414;padding:2px 9px;cursor:pointer;font-family:inherit;font-size:11px;color:#aaa;transition:border-color .15s,color .15s;}
.mchip:hover:not(:disabled){border-color:#666;color:#eee;}
.mchip:disabled{opacity:.35;cursor:not-allowed;}
.mchip .mlabel{color:#666;font-size:10px;letter-spacing:.05em;}
.mchip .mval{color:#88ccff;}
.mchip .marr{color:#555;font-size:10px;}
.mchip.judge-chip .mval{color:#f0c060;}
.magent{display:inline-flex;align-items:center;gap:5px;border:1px solid #2a2a2a;background:#0e0e0e;padding:2px 9px;font-family:inherit;font-size:11px;color:#888;}
.magent .mlabel{color:#555;font-size:10px;letter-spacing:.05em;}
.magent .mval{color:#70ddaa;}

/* modal overlay for model selection */
.mmod{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:200;align-items:center;justify-content:center;}
.mmod.open{display:flex;}
.mmodbox{background:#161616;border:1px solid #3a3a3a;min-width:300px;}
.mmodhdr{padding:8px 14px;border-bottom:1px solid #252525;display:flex;justify-content:space-between;align-items:center;}
.mmodttl{color:#ccc;font-size:13px;}
.mmodcls{color:#666;cursor:pointer;font-size:18px;line-height:1;}
.mmodcls:hover{color:#eee;}
.mmodlist{padding:6px 0;}
.mmodopt{padding:9px 16px;cursor:pointer;font-size:13px;color:#aaa;border-left:3px solid transparent;}
.mmodopt:hover{background:#1e1e1e;color:#eee;}
.mmodopt.sel{border-left-color:#88ccff;color:#eee;background:#1a1a1a;}
.mmodopt .mok{font-size:11px;color:#888;margin-left:8px;}

/* buttons */
.btn{border:1px solid #484848;font-family:inherit;font-size:12px;padding:3px 13px;cursor:pointer;background:transparent;color:#aaa;}
.btn:hover{border-color:#888;color:#eee;}
.btn:disabled{opacity:.28;cursor:not-allowed;}
.btn-s{border-color:#3a6a3a;color:#6ee886;background:#0d1a0d;}
.btn-s:not(:disabled):hover{background:#1a2e1a;}
.btn-x{border-color:#6a2828;color:#e07070;background:#1a0d0d;}
.btn-x:not(:disabled):hover{background:#2e1616;}

/* layout */
/* CRITICAL: .left and .right need explicit min-height:0 + overflow:hidden so
   their flex children (.det, .cbody) can scroll. Grid items default to
   min-height:auto and grow with content - that defeats overflow:auto on
   inner panels. Do NOT remove these without testing scroll behavior. */
.body{flex:1;display:grid;grid-template-columns:440px 1fr;min-height:0;overflow:hidden;}

/* left panel */
.left{border-right:2px solid #242424;display:flex;flex-direction:column;overflow:hidden;min-height:0;}
.sh{background:#161616;border-bottom:1px solid #242424;padding:4px 13px;display:flex;justify-content:space-between;align-items:center;flex-shrink:0;}
.shl{color:#999;font-size:11px;letter-spacing:.05em;}
.shr{color:#888;font-size:11px;}

/* queue */
.qwrap{flex:0 0 auto;overflow-y:auto;max-height:210px;}
.qcols{display:grid;grid-template-columns:24px 1fr 72px 44px;padding:3px 13px;color:#666;font-size:11px;border-bottom:1px solid #1e1e1e;}
.qrow{display:grid;grid-template-columns:24px 1fr 72px 44px;padding:6px 13px;border-bottom:1px solid #1e1e1e;align-items:center;cursor:pointer;border-left:3px solid #282828;}
.qrow:hover{background:#181818;}
.qrow.sel{background:#1a1a1a;}
.sp{border-left-color:#1D9E75 !important;}
.sr{border-left-color:#EF9F27 !important;background:#141208 !important;}
.se{border-left-color:#cc4444 !important;}
.si{border-left-color:#777 !important;}
.so{border-left-color:#333 !important;}
.qnum{color:#666;font-size:12px;}
.qnm{color:#eee;font-size:13px;}
.qnmd{color:#aaa;font-size:13px;}
.qat{color:#888;font-size:11px;}
.ql{font-size:12px;}
.lp{color:#3dd898;}
.lr{color:#f0a830;}
.le{color:#e05858;}
.li{color:#aaa;}
.lo{color:#777;}
.blink{animation:blink .6s step-end infinite;}
@keyframes blink{50%{opacity:0;}}

/* detail */
.dsh{background:#161616;border-top:2px solid #242424;border-bottom:1px solid #242424;padding:4px 13px;flex-shrink:0;}
.dsh span{color:#999;font-size:11px;letter-spacing:.04em;}
.det{flex:1;overflow-y:auto;padding:11px 15px 18px;}
.demp{color:#666;font-size:13px;}
.dl{color:#aaa;font-size:11px;letter-spacing:.07em;margin-top:12px;margin-bottom:4px;}
.dl:first-child{margin-top:0;}
.dv{color:#ccc;font-size:12px;line-height:1.55;word-break:break-word;}
.da{display:flex;gap:6px;flex-wrap:wrap;margin-top:13px;}
.db{border:1px solid #444;color:#aaa;font-size:11px;padding:3px 10px;font-family:inherit;cursor:pointer;background:transparent;}
.db:hover{border-color:#888;color:#eee;}
.dbr{border-color:#7a4a00 !important;color:#f0b030 !important;}
.dbr:hover{background:#1e1200 !important;}
.dbp{border-color:#2a5a30 !important;color:#50cc70 !important;}
.dbp:hover{background:#102018 !important;}
.vb{margin-top:11px;border-top:1px solid #242424;padding-top:9px;font-size:12px;line-height:1.75;}
.vk{color:#aaa;}
.vp{color:#3dd898;}
.vf{color:#e05858;}
.ve{color:#f0a830;}
.vv{color:#ccc;}
.vi{color:#e09030;display:block;}
.vfb{color:#80c880;display:block;}
.vesc{color:#e05858;display:block;}

/* console */
.right{display:flex;flex-direction:column;background:#0c0c0c;overflow:hidden;min-height:0;}
.ch{background:#101010;border-bottom:1px solid #1e1e1e;padding:4px 13px;display:flex;justify-content:space-between;flex-shrink:0;}
.ch span{color:#777;font-size:11px;}
.chtask{color:#aaa !important;}
.cbody{flex:1;overflow-y:auto;padding:8px 16px 12px;min-height:0;}
.cbody div{font-size:13px;line-height:1.5;white-space:pre-wrap;word-break:break-all;min-height:1.2em;}
.cn{color:#60d878;}
.ch2{color:#90eeaa;}
.cd{color:#5a8a68;}
.ca{color:#f0b030;}
.cr{color:#e06868;}
.cb{color:#70aadd;}
.cw{color:#ddd;}
.cs{color:#2a4a36;}

/* footer */
.foot{background:#0e0e0e;border-top:1px solid #1e1e1e;padding:4px 16px;display:flex;justify-content:space-between;color:#666;font-size:11px;flex-shrink:0;}

/* edit modal */
.mbg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:100;align-items:center;justify-content:center;}
.mbg.open{display:flex;}
.mbox{background:#141414;border:1px solid #383838;width:640px;max-height:80vh;display:flex;flex-direction:column;}
.mhdr{padding:8px 15px;border-bottom:1px solid #282828;display:flex;justify-content:space-between;align-items:center;}
.mttl{color:#ccc;font-size:13px;}
.mcls{color:#777;cursor:pointer;font-size:18px;line-height:1;}
.mcls:hover{color:#eee;}
.mbdy{flex:1;padding:9px;display:flex;flex-direction:column;overflow:hidden;}
.mbdy textarea{flex:1;min-height:280px;background:#0a0a0a;color:#ccc;border:1px solid #2a2a2a;font-family:inherit;font-size:13px;padding:9px;resize:none;outline:none;line-height:1.55;}
.mbdy textarea:focus{border-color:#484848;}
.mft{padding:8px 9px;display:flex;gap:7px;justify-content:flex-end;border-top:1px solid #1e1e1e;}
.mb{border:1px solid #383838;color:#aaa;font-size:12px;padding:3px 13px;font-family:inherit;cursor:pointer;background:transparent;}
.mb:hover{border-color:#777;color:#eee;}
.mbs{border-color:#3a6a3a !important;color:#6ee886 !important;background:#0d1a0d !important;}
.mbs:hover{background:#1a2e1a !important;}

::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:#0c0c0c;}
::-webkit-scrollbar-thumb{background:#2e2e2e;}
::-webkit-scrollbar-thumb:hover{background:#444;}
</style>
</head>
<body>
<div class="app">

  <div class="tb">
    <span class="tname">PQ_MINDER</span>
    <div class="tctrl">
      <button class="btn btn-s" id="btn-start" onclick="doStart()">START</button>
      <button class="btn btn-x" id="btn-stop" onclick="doStop()" disabled>STOP</button>
      <span class="tstat" id="tstat">IDLE</span>
    </div>
    <div class="tmodels">
      <span class="tmsep">|</span>
      <button class="mchip judge-chip" id="chip-judge" onclick="openJudgeModal()" title="Click to change judge model">
        <span class="mlabel">JUDGE</span>
        <span class="mval" id="chip-judge-val">&mdash;</span>
        <span class="marr">&#9662;</span>
      </button>
      <span class="magent" title="Agent model is fixed">
        <span class="mlabel">AGENT</span>
        <span class="mval" id="chip-agent-val">&mdash;</span>
      </span>
    </div>
  </div>

  <div class="body">
    <div class="left">
      <div class="sh">
        <span class="shl" id="qcount">QUEUE [0]</span>
        <span class="shr" id="qsum"></span>
      </div>
      <div class="qcols"><span>#</span><span>TASK_ID</span><span>STATUS</span><span>ATT</span></div>
      <div class="qwrap" id="qlist"></div>
      <div class="dsh"><span id="dsht">SELECTED: &mdash;</span></div>
      <div class="det" id="det"><div class="demp">click a task to view details</div></div>
    </div>

    <div class="right">
      <div class="ch">
        <span>CONSOLE [LIVE]</span>
        <span class="chtask" id="ctask">&mdash;</span>
      </div>
      <div class="cbody" id="cbody"></div>
    </div>
  </div>

  <div class="foot">
    <span id="fagent">agent: &mdash;</span>
    <span id="fjudge">judge: &mdash;</span>
    <span id="felapsed">elapsed: &mdash;</span>
    <span id="flines">lines: 0</span>
  </div>
</div>

<!-- judge selection modal -->
<div class="mmod" id="mmod" onclick="if(event.target===this)closeJudgeModal()">
  <div class="mmodbox">
    <div class="mmodhdr">
      <span class="mmodttl">SELECT JUDGE PROVIDER</span>
      <span class="mmodcls" onclick="closeJudgeModal()">&#xd7;</span>
    </div>
    <div class="mmodlist" id="mmodlist"></div>
  </div>
</div>

<!-- p/q edit modal -->
<div class="mbg" id="modal" onclick="if(event.target===this)closeModal()">
  <div class="mbox">
    <div class="mhdr">
      <span class="mttl" id="mttl">EDIT</span>
      <span class="mcls" onclick="closeModal()">&#xd7;</span>
    </div>
    <div class="mbdy"><textarea id="mta" spellcheck="false"></textarea></div>
    <div class="mft">
      <button class="mb" onclick="closeModal()">CANCEL</button>
      <button class="mb mbs" onclick="saveModal()">SAVE</button>
    </div>
  </div>
</div>

<script>
var selTask=null, lastState=null, modalUrl=null, es=null, lineCount=0;
var judgeOptions=[], curJudge='';
var agentLabel='';

// followTail: are we tracking the bottom of the console like a real `tail -f`?
// Starts true. Flips to false when the user scrolls UP away from the bottom.
// Flips back to true when the user scrolls back to within FOLLOW_THRESHOLD of bottom.
// programmaticScrollPending: set when WE programmatically scroll to bottom, so the
// resulting 'scroll' event doesn't get treated as a user action.
var followTail=true;
var programmaticScrollPending=false;
var FOLLOW_THRESHOLD=60; // px from bottom that still counts as "at bottom"

var SLBL={open:'[PEND]',passed:'[PASS]',failed:'[FAIL]',escalated:'[ESC!]',interrupted:'[INT]',blocked:'[BLKD]',stopping:'[STOP]'};
var SCSS={open:'so',passed:'sp',failed:'se',escalated:'se',interrupted:'si',blocked:'so',stopping:'si'};
var SLC={open:'lo',passed:'lp',failed:'le',escalated:'le',interrupted:'li',blocked:'lo',stopping:'li'};

function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function pad(n){return n<10?'0'+n:''+n;}
function fmtT(s){var h=Math.floor(s/3600),m=Math.floor((s%3600)/60),ss=s%60;return(h?pad(h)+':':'')+pad(m)+':'+pad(ss);}

function colorLine(r){
  if(!r||!r.trim()) return '<span class="cs">&nbsp;</span>';
  var t=esc(r);
  if(/\[tool call\]/i.test(r))    return '<span class="ca">'+t+'</span>';
  if(/\[escalate\]/i.test(r)||/\[ERROR\]/i.test(r)) return '<span class="cr">'+t+'</span>';
  if(/\bPASSED\b/.test(r)||/passed on attempt/i.test(r)||/All tasks complete/i.test(r)) return '<span class="ch2">'+t+'</span>';
  if(/\[warn\]/i.test(r)||/\[ESCALATE\]/i.test(r)||/\[STOPPED\]/i.test(r)||/low-confidence/i.test(r)) return '<span class="ca">'+t+'</span>';
  if(/calling judge/i.test(r)||/JUDGE RESPONSE/i.test(r)||/JUDGE PROMPT/i.test(r)) return '<span class="cb">'+t+'</span>';
  if(/^Verdict:/i.test(r)) return '<span class="cw">'+t+'</span>';
  if(/^Issues:/i.test(r)) return '<span class="ca">'+t+'</span>';
  if(/^Task\s+\S+\s+attempt/i.test(r)) return '<span class="ch2">'+t+'</span>';
  if(/\[pq_minder\]/.test(r)||/^Starting agent/.test(r)||/^MCP/.test(r)||/^Agent model/.test(r)||/^Queue:/.test(r)||/^Judge:/.test(r)||/^Agent:/.test(r)) return '<span class="cd">'+t+'</span>';
  return '<span class="cn">'+t+'</span>';
}

function appendLine(raw){
  var cb=document.getElementById('cbody');
  var d=document.createElement('div');
  d.innerHTML=colorLine(raw);
  cb.appendChild(d);
  lineCount++;
  document.getElementById('flines').textContent='lines: '+lineCount;
  // tail-follow: snap to the new bottom on the next frame so layout has settled.
  // setting scrollTop fires a 'scroll' event - the flag tells the listener to ignore it.
  if(followTail){
    programmaticScrollPending=true;
    requestAnimationFrame(function(){
      cb.scrollTop=cb.scrollHeight;
    });
  }
}

// distinguish user scroll from programmatic scroll: programmatic scrolls set the
// flag, which the listener consumes; user wheel/touch/drag falls through and
// re-evaluates whether to keep tailing
function setupConsoleScroll(){
  var cb=document.getElementById('cbody');
  cb.addEventListener('scroll',function(){
    if(programmaticScrollPending){programmaticScrollPending=false;return;}
    var distFromBot=cb.scrollHeight-cb.scrollTop-cb.clientHeight;
    followTail=(distFromBot < FOLLOW_THRESHOLD);
  });
}

function connectSSE(){
  if(es){es.close();es=null;}
  es=new EventSource('/api/console/stream');
  es.onmessage=function(e){appendLine(JSON.parse(e.data));};
  es.onerror=function(){};
}

function pollState(){
  fetch('/api/state').then(function(r){return r.json();}).then(function(s){
    lastState=s;
    renderStatus(s);
    renderQueue(s);
    renderFooter(s);
    renderModelChips(s);
    if(selTask){var t=s.tasks.find(function(x){return x.id===selTask;});if(t)renderDetail(t,s);}
  }).catch(function(){});
}

function fetchJudgeOptions(){
  fetch('/api/judge').then(function(r){return r.json();}).then(function(d){
    judgeOptions=d.judge_options||[];
    curJudge=d.judge||'';
    updateChipLabels();
  });
}

function labelForJudge(key){
  for(var i=0;i<judgeOptions.length;i++){if(judgeOptions[i].key===key)return judgeOptions[i].label;}
  return key||'';
}

function updateChipLabels(){
  document.getElementById('chip-judge-val').textContent=labelForJudge(curJudge)||'\u2014';
  document.getElementById('chip-agent-val').textContent=agentLabel||'\u2014';
}

function renderModelChips(s){
  var running=(s.runner_status==='running'||s.runner_status==='stopping');
  document.getElementById('chip-judge').disabled=running;
  if(s.judge_provider&&s.judge_provider!==curJudge){curJudge=s.judge_provider;}
  if(s.agent_model){
    // Display a friendly short label - last segment of slug
    agentLabel=s.agent_model.split('/').pop();
  }
  updateChipLabels();
}

function openJudgeModal(){
  if(lastState&&(lastState.runner_status==='running'||lastState.runner_status==='stopping'))return;
  document.getElementById('mmodlist').innerHTML='';
  var html='';
  judgeOptions.forEach(function(o){
    var isSel=(o.key===curJudge);
    html+='<div class="mmodopt'+(isSel?' sel':'')+'" data-key="'+esc(o.key)+'" onclick="selectJudge(\''+esc(o.key)+'\')">';
    html+=esc(o.label);
    html+='<span class="mok">'+esc(o.key)+'</span>';
    html+='</div>';
  });
  document.getElementById('mmodlist').innerHTML=html;
  document.getElementById('mmod').classList.add('open');
}

function closeJudgeModal(){
  document.getElementById('mmod').classList.remove('open');
}

function selectJudge(key){
  fetch('/api/judge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({judge:key})})
    .then(function(r){return r.json();})
    .then(function(d){
      if(d.error){alert(d.error);return;}
      curJudge=d.judge;
      updateChipLabels();
      closeJudgeModal();
    });
}

function renderStatus(s){
  var rs=s.runner_status,tasks=s.tasks||[];
  var np=tasks.filter(function(t){return t.status==='passed';}).length;
  var ne=tasks.filter(function(t){return t.status==='escalated';}).length;
  var lbl='';
  if(rs==='running')       lbl='<span style="color:#f0b030">RUNNING</span>';
  else if(rs==='stopping') lbl='<span style="color:#e07070">STOPPING</span>';
  else if(rs==='stopped')  lbl=s.error?'<span style="color:#e07070">ERROR</span>':'<span style="color:#888">STOPPED</span>';
  else                     lbl='<span style="color:#777">IDLE</span>';
  if(s.current_task&&rs==='running')
    lbl+=' | <span style="color:#bbb">'+esc(s.current_task)+(s.current_run?' / '+esc(s.current_run):'')+'</span>';
  lbl+=' &nbsp;|&nbsp; TASKS:'+tasks.length+' PASSED:'+np;
  if(ne) lbl+=' ESC:<span style="color:#e07070">'+ne+'</span>';
  document.getElementById('tstat').innerHTML=lbl;
  document.getElementById('btn-start').disabled=(rs==='running'||rs==='stopping');
  document.getElementById('btn-stop').disabled=(rs!=='running');
  document.getElementById('ctask').textContent=
    (s.current_task&&rs==='running')?(s.current_task+(s.current_run?' / '+s.current_run:'')):'\u2014';
}

function renderQueue(s){
  var tasks=s.tasks||[], html='';
  tasks.forEach(function(t){
    var isRun=(s.current_task===t.id&&s.runner_status==='running');
    var sc=isRun?'sr':(SCSS[t.status]||'so');
    var lc=isRun?'lr':(SLC[t.status]||'lo');
    var lbl=isRun?'[RUN]':(SLBL[t.status]||'[???]');
    var nc=(t.status==='passed')?'qnm':'qnmd';
    var sel=(selTask===t.id)?' sel':'';
    html+='<div class="qrow '+sc+sel+'" data-id="'+esc(t.id)+'">';
    html+='<span class="qnum">'+t.index+'</span>';
    html+='<span class="'+nc+'">'+esc(t.id)+'</span>';
    html+='<span class="ql '+lc+'">'+lbl+(isRun?'<span class="blink"> &#9646;</span>':'')+'</span>';
    html+='<span class="qat">'+t.attempts+'/'+(s.max_attempts||3)+'</span>';
    html+='</div>';
  });
  var el=document.getElementById('qlist');
  el.innerHTML=html;
  el.querySelectorAll('.qrow').forEach(function(r){
    r.addEventListener('click',function(){selectTask(this.dataset.id);});
  });
  document.getElementById('qcount').textContent='QUEUE ['+tasks.length+']';
  var np=tasks.filter(function(t){return t.status==='passed';}).length;
  var ne=tasks.filter(function(t){return t.status==='escalated';}).length;
  var sum=np+'/'+tasks.length+' PASSED';
  if(ne) sum+=' | '+ne+' ESC';
  document.getElementById('qsum').textContent=sum;
}

function selectTask(id){
  selTask=id;
  if(!lastState) return;
  var t=lastState.tasks.find(function(x){return x.id===id;});
  if(t) renderDetail(t,lastState);
}

function renderDetail(t,s){
  var isRun=(s.current_task===t.id&&s.runner_status==='running');
  document.getElementById('dsht').textContent='SELECTED: '+t.id;
  var h='';
  h+='<div class="dl">P.MD</div>';
  h+='<div class="dv">'+esc(t.p_text.length>240?t.p_text.substring(0,240)+'\u2026':t.p_text)+'</div>';
  h+='<div class="dl">Q.MD</div>';
  h+='<div class="dv">'+esc(t.q_text.length>170?t.q_text.substring(0,170)+'\u2026':t.q_text)+'</div>';
  h+='<div class="da">';
  h+='<button class="db" onclick="editFile(\''+t.id+'\',\'p\')">[EDIT P]</button>';
  h+='<button class="db" onclick="editFile(\''+t.id+'\',\'q\')">[EDIT Q]</button>';
  if(!isRun){
    h+='<button class="db dbr" onclick="doRetry(\''+t.id+'\')">[RETRY]</button>';
    if(t.status!=='passed')
      h+='<button class="db dbp" onclick="doPass(\''+t.id+'\')">[FLAG:PASS]</button>';
  }
  h+='</div>';
  var v=t.last_verdict;
  if(v&&v.status){
    var vc=(v.status==='passed')?'vp':(v.escalation_reason?'ve':'vf');
    h+='<div class="vb">';
    h+='<span class="vk">VERDICT: </span><span class="'+vc+'">'+v.status.toUpperCase()+'</span>';
    if(v.confidence!==undefined) h+=' <span class="vk">conf=</span><span class="vv">'+parseFloat(v.confidence).toFixed(2)+'</span>';
    if(v.agent_model) h+=' <span class="vk">agent=</span><span class="vv">'+esc(v.agent_model.split('/').pop())+'</span>';
    if(v.judge_provider) h+=' <span class="vk">judge=</span><span class="vv">'+esc(v.judge_provider)+'</span>';
    if(v.elapsed_s)   h+=' <span class="vk">t=</span><span class="vv">'+fmtT(v.elapsed_s)+'</span>';
    if(v.issues&&v.issues.length) v.issues.forEach(function(i){h+='<span class="vi">&#8627; '+esc(i)+'</span>';});
    if(v.feedback) h+='<span class="vfb">feedback: '+esc(v.feedback)+'</span>';
    if(v.escalation_reason) h+='<span class="vesc">esc: '+esc(v.escalation_reason)+'</span>';
    h+='</div>';
  }
  document.getElementById('det').innerHTML=h;
}

function renderFooter(s){
  document.getElementById('fagent').textContent='agent: '+(s.agent_model||'\u2014');
  document.getElementById('fjudge').textContent='judge: '+(s.judge_provider||'\u2014');
  document.getElementById('felapsed').textContent=
    (s.runner_status==='running'&&s.elapsed)?'elapsed: '+fmtT(s.elapsed):'elapsed: \u2014';
}

function doStart(){
  fetch('/api/start',{method:'POST'}).then(function(){
    document.getElementById('cbody').innerHTML='';
    lineCount=0;
    document.getElementById('flines').textContent='lines: 0';
    // fresh run: re-engage tail-follow
    followTail=true;
    connectSSE();
    pollState();
  });
}
function doStop(){fetch('/api/stop',{method:'POST'}).then(pollState);}

function doRetry(id){
  if(!confirm('Reset "'+id+'" to open (attempts=0)?')) return;
  fetch('/api/task/'+id+'/retry',{method:'POST'}).then(pollState);
}
function doPass(id){
  if(!confirm('Mark "'+id+'" as PASSED?')) return;
  fetch('/api/task/'+id+'/pass',{method:'POST'}).then(pollState);
}

function editFile(id,type){
  modalUrl='/api/task/'+id+'/'+type;
  document.getElementById('mttl').textContent='EDIT '+type.toUpperCase()+'.MD \u2014 '+id;
  fetch('/api/task/'+id+'/'+type).then(function(r){return r.json();}).then(function(d){
    document.getElementById('mta').value=d.content||'';
    document.getElementById('modal').classList.add('open');
    setTimeout(function(){document.getElementById('mta').focus();},40);
  });
}
function closeModal(){document.getElementById('modal').classList.remove('open');}
function saveModal(){
  var c=document.getElementById('mta').value;
  fetch(modalUrl,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content:c})})
    .then(function(){closeModal();pollState();});
}

document.addEventListener('keydown',function(e){
  if(e.key==='Escape'){closeModal();closeJudgeModal();}
});

// startup
setupConsoleScroll();
setInterval(pollState, 2000);
pollState();
connectSSE();
fetchJudgeOptions();
</script>
</body>
</html>"""
