/* Recorded measurements only. This script never starts a model benchmark. */
(() => {
  const $ = id => document.getElementById(id);
  const base = '#97a697', candidate = '#1e6b57';
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = x => x.toFixed(2);
  const text = (x,y,s,cls='',anchor='start') => `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`;
  const line = (x1,y1,x2,y2,cls='grid',extra='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${cls}" ${extra}/>`;
  const svg = (height, title, contents) => `<svg viewBox="0 0 1000 ${height}" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title>${contents}</svg>`;
  const scale = (v,max,left=190,width=685) => left + v/max*width;
  function paired(rows,max,unit) {
    const height=rows.length*87+68; let content='';
    for(let t=0;t<=4;t++) {const v=t*max/4,x=scale(v,max);content+=line(x,16,x,height-35)+text(x,height-13,`${v}${unit}`,'axis-label','middle');}
    rows.forEach((r,i)=>{const y=31+i*87;content+=text(0,y+14,r.name,'row-label');
      [r.baseline,r.candidate].forEach((v,j)=>{const yy=y+j*26;content+=`<rect x="190" y="${yy}" width="${scale(v,max)-190}" height="17" fill="${j?candidate:base}"/>`+text(scale(v,max)+9,yy+13,`${fmt(v)}${unit}`,'value');});});
    return svg(height,'Baseline and candidate comparison; all bars start at zero',content);
  }
  async function start() {
    const [heldout,data] = await Promise.all(['./evidence/testdev-results.json','./evidence/chart-data.json'].map(async url=>{const r=await fetch(url);if(!r.ok)throw new Error('Recorded result could not load');return r.json();}));
    const a=heldout.comparisons['whole-640'].modes.whole,b=heldout.comparisons['whole-1280'].modes.whole;
    const fixed = [
      {name:'Precision',baseline:69.93073665909537,candidate:69.88518577579333},
      {name:'Recall',baseline:33.205507176906074,candidate:46.68317754520519},
      {name:'Micro F1',baseline:45.029477352545527,candidate:55.97509379739762}
    ];
    function heldoutChart(){
      const mode=$('heldout-metric').value;
      const rows=mode==='accuracy'?[
        {name:'Overall AP',baseline:a.ap*100,candidate:b.ap*100},
        {name:'Small-object AP',baseline:a.ap_small*100,candidate:b.ap_small*100},
        {name:'Small-object recall',baseline:a.recall_small*100,candidate:b.recall_small*100}
      ]:mode==='operating'?fixed:[
        {name:'Median latency',baseline:a.latency_p50_ms,candidate:b.latency_p50_ms},
        {name:'p95 latency',baseline:a.latency_p95_ms,candidate:b.latency_p95_ms}
      ];
      $('heldout-chart').innerHTML=paired(rows,mode==='accuracy'?30:mode==='operating'?100:30,mode==='latency'?' ms':'%');
      $('heldout-caption').textContent=mode==='accuracy'?'Custom COCO-style AP and small-object average recall, IoU 0.50–0.95, using confidence 0.001 for integration. Shared 0–30% scale. These are not official VisDrone challenge scores.':mode==='operating'?'Confidence fixed at 0.30 before held-out evaluation; same-class matching at IoU 0.5. Shared 0–100% scale. These metrics differ from AP.':'Batch-one PyTorch on Tesla T4, with warmup and synchronization; includes preprocessing, inference, postprocessing and transfer, excludes image-file decoding. Lower latency is better. Shared 0–30 ms scale. Within-run quantiles, not confidence intervals.';
    }
    let selected=2;
    function strategyChart(){
      const metric=$('strategy-metric').value, label={ap:'Overall AP (%)',small:'Small-object AP (%)',recall:'Small-object average recall (%)'}[metric];
      const x=v=>90+v/80*820,y=v=>300-v/35*250;
      let content='';
      for(let v=0;v<=80;v+=20)content+=line(x(v),50,x(v),300)+text(x(v),326,v,'axis-label','middle');
      for(let v=0;v<=35;v+=5)content+=line(90,y(v),910,y(v))+text(74,y(v)+4,v,'axis-label','end');
      content+=text(90,22,label,'axis-label')+text(500,362,'Median T4 latency (ms) · lower is better','axis-label','middle');
      data.validation.forEach((r,i)=>{const xx=x(r.latency),yy=y(r[metric]),offsets=[[-12,24,'end'],[12,8,'start'],[12,-13,'start'],[12,-12,'start']],o=offsets[i];content+=`<circle cx="${xx}" cy="${yy}" r="${i===selected?9:6}" fill="${i===selected?candidate:base}" stroke="#fcfcf7" stroke-width="2"><title>${esc(r.name)}: ${r[metric]}%, ${r.latency} ms</title></circle>`+text(xx+o[0],yy+o[1],r.name,'row-label',o[2]);});
      $('strategy-chart').innerHTML=svg(380,`${label} versus median latency on 548 validation images`,content);
      const r=data.validation[selected];
      $('strategy-detail').textContent=`${r.name}: ${fmt(r[metric])}% · ${r.latency.toFixed(1)} ms. Whole 640 baseline: ${fmt(data.validation[0][metric])}% · 12.5 ms.`;
      document.querySelectorAll('[data-strategy]').forEach(el=>el.setAttribute('aria-pressed',String(Number(el.dataset.strategy)===selected)));
    }
    function classChart(){
      const rows=[...data.classes].sort($('class-order').value==='gain'?(a,b)=>(b.candidate-b.baseline)-(a.candidate-a.baseline):(a,b)=>b.candidate-a.candidate);
      const x=v=>185+v/60*630;let content='';
      for(let v=0;v<=60;v+=10)content+=line(x(v),25,x(v),465)+text(x(v),491,`${v}%`,'axis-label','middle');
      content+=text(958,16,'640 → 1280','axis-label','end');
      rows.forEach((r,i)=>{const y=42+i*44;content+=text(0,y+4,r.name==='motor'?'Motorbike':r.name==='awning-tricycle'?'Covered tricycle':r.name[0].toUpperCase()+r.name.slice(1),'row-label')+line(x(r.baseline),y,x(r.candidate),y,'',`stroke="#b6c1b0" stroke-width="2"`)+`<circle cx="${x(r.baseline)}" cy="${y}" r="4" fill="${base}"/><circle cx="${x(r.candidate)}" cy="${y}" r="5" fill="${candidate}"/>`+text(977,y+4,`${fmt(r.baseline)} → ${fmt(r.candidate)}`,'value','end');});
      $('class-chart').innerHTML=svg(508,'Per-class held-out AP, whole 640 baseline versus whole 1280 candidate',content);
    }
    $('heldout-metric').addEventListener('change',heldoutChart);$('strategy-metric').addEventListener('change',strategyChart);$('class-order').addEventListener('change',classChart);
    document.querySelectorAll('[data-strategy]').forEach(el=>el.addEventListener('click',()=>{selected=Number(el.dataset.strategy);strategyChart();}));
    heldoutChart();strategyChart();classChart();
  }
  start().catch(()=>document.querySelectorAll('.chart').forEach(el=>{el.textContent='Charts could not load. Download the original experiment records below to inspect the results.';}));
})();
