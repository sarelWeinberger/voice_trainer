/* Read-only plot of the already saved scores. No storage or API writes. */
(function(root){
    'use strict';
    const titles={G:'דירוג כללי',R:'צרידות',B:'נשיפות',A:'חולשה',S:'מאמץ קולי',posture:'תנוחה ותנועה',neck:'מראה הצוואר',jaw:'תנועת לסת'};
    const colors=['#2563eb','#d97706','#9333ea','#059669','#dc2626'];
    function smoothHistory(values,versions=[]){
        // Causal exponential mean: past points never change when a new result arrives.
        // Reset at missing values and rubric/model boundaries; never fill missing grades.
        let previous=null,version;
        return values.map((value,index)=>{
            if(value===null || !Number.isFinite(value)){previous=null;version=undefined;return null;}
            if(previous===null || versions[index]!==version)previous=value;
            else previous=.35*value+.65*previous;
            version=versions[index];
            return previous;
        });
    }
    function historySeries(entries,kind='neck'){
        const rows=(Array.isArray(entries)?entries:[]).filter(entry=>Number.isFinite(Date.parse(entry.timestamp||entry.date)))
            .slice().sort((a,b)=>Date.parse(b.timestamp||b.date)-Date.parse(a.timestamp||a.date)).slice(0,10).reverse();
        const keys=kind==='voice'?['G','R','B','A','S']:['posture','neck','jaw'];
        const result={rows,series:keys.map((key,index)=>({key,title:titles[key],color:colors[index],values:rows.map(entry=>{
            if(kind==='voice'){
                const value=entry.scores?.[key];return Number.isFinite(value)&&value>=0&&value<=3?value:null;
            }
            const group=root.neckGroupResults(entry.neck_effort).find(group=>group.key===key);
            return group && !group.partial?group.score:null;
        })}))};
        const versions=kind==='neck'?rows.map(entry=>JSON.stringify(Object.entries(entry.neck_effort?.neckAnalysis?.versions||{}).sort())):[];
        result.series.forEach(line=>{line.trend=smoothHistory(line.values,versions);});
        return result;
    }
    function renderHistoryProgress(container,entries){
        if(!container)return;
        const kind=container.getAttribute('data-score-kind')||'neck';
        container.replaceChildren();
        const heading=document.createElement('h3');heading.textContent='גרף השיפור שלי';container.appendChild(heading);
        const controls=document.createElement('div');controls.className='history-progress-controls';
        for(const [value,title] of [['neck','צוואר'],['voice','קול · GRBAS']]){
            const button=document.createElement('button');button.type='button';button.textContent=title;
            button.setAttribute('aria-pressed',String(kind===value));
            button.onclick=()=>{container.setAttribute('data-score-kind',value);renderHistoryProgress(container,entries);};
            controls.appendChild(button);
        }
        container.appendChild(controls);
        const {rows,series}=historySeries(entries,kind);
        const note=document.createElement('p');
        note.textContent='עשר ההקלטות האחרונות, מהישנה לחדשה. 0 הוא הדירוג הרצוי; 3 הוא דירוג גבוה ופחות רצוי. ירידה בציון היא כיוון השיפור — פחות סימנים באותו מדד. נתון חסר אינו ציון 0.';
        container.appendChild(note);
        const trendNote=document.createElement('p');trendNote.textContent='הקו מציג מגמה מוחלקת לאורך ההקלטות; הנקודות הן הציונים המקוריים. הקו מגיב בהדרגה גם לשיפור וגם לעלייה בציונים.';container.appendChild(trendNote);
        if(!rows.length){const empty=document.createElement('p');empty.textContent='הגרף יופיע לאחר שתישמר תוצאה בהיסטוריה.';container.appendChild(empty);return;}
        const ns='http://www.w3.org/2000/svg';
        const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 640 270');svg.setAttribute('role','img');
        svg.setAttribute('aria-label','ציוני '+(kind==='voice'?'GRBAS':'הצוואר')+' בהקלטות האחרונות. הנתונים המלאים בטבלה שמתחת לגרף.');
        svg.style.width='100%';svg.style.display='block';svg.setAttribute('dir','ltr');
        const add=(name,attrs,text)=>{const node=document.createElementNS(ns,name);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value));if(text!==undefined)node.textContent=text;svg.appendChild(node);return node;};
        for(let grade=0;grade<=3;grade++){
            const y=220-grade*64;add('line',{x1:40,x2:610,y1:y,y2:y,stroke:'#cbd5e1'});add('text',{x:20,y:y+5,fill:'currentColor','font-size':15},grade);
        }
        const x=index=>rows.length===1?325:50+index*550/(rows.length-1);
        rows.forEach((entry,index)=>add('text',{x:x(index),y:248,fill:'currentColor','font-size':14,'text-anchor':'middle'},index+1));
        const legend=document.createElement('div');legend.className='history-progress-legend';
        series.forEach(line=>{
            const label=document.createElement('span');label.style.color=line.color;label.textContent='● '+line.title;legend.appendChild(label);
            let previous=null;
            line.values.forEach((value,index)=>{
                if(value===null){previous=null;return;}
                const point={x:x(index),y:220-line.trend[index]*64};
                const version=JSON.stringify(Object.entries(rows[index].neck_effort?.neckAnalysis?.versions||{}).sort());
                if(previous && (kind==='voice'||previous.version===version))add('line',{x1:previous.x,y1:previous.y,x2:point.x,y2:point.y,stroke:line.color,'stroke-width':2.5});
                const dot=add('circle',{cx:point.x,cy:220-value*64,r:3,fill:line.color,opacity:.5});
                const tooltip=document.createElementNS(ns,'title');tooltip.textContent=`${line.title}: ${value} · ${rows[index].timestamp||rows[index].date}`;dot.appendChild(tooltip);
                previous={...point,version};
            });
        });
        container.appendChild(legend);container.appendChild(svg);
        if(!series.some(line=>line.values.some(value=>value!==null))){const p=document.createElement('p');p.textContent='אין בהקלטות האלה ציונים מלאים להצגה בקבוצה שנבחרה.';container.appendChild(p);}
        const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='הצגת התאריכים והציונים';details.appendChild(summary);
        const method=document.createElement('p');method.textContent='קו המגמה הוא ממוצע נע מעריכי: 35% לציון החדש ו־65% למגמה הקודמת. הציונים השמורים אינם משתנים. לאחר נתון חסר או שינוי גרסת ניתוח הצוואר מתחיל רצף חדש.';details.appendChild(method);
        const table=document.createElement('table');table.className='neck-change-table';
        const head=table.insertRow();['הקלטה','תאריך',...series.map(line=>line.title)].forEach(text=>{const th=document.createElement('th');th.textContent=text;head.appendChild(th);});
        rows.forEach((entry,index)=>{const row=table.insertRow();[index+1,new Date(entry.timestamp||entry.date).toLocaleString('he-IL'),...series.map(line=>line.values[index]??'—')].forEach(text=>{row.insertCell().textContent=text;});});
        details.appendChild(table);container.appendChild(details);
    }
    root.renderHistoryProgress=renderHistoryProgress;
    if(typeof module!=='undefined')module.exports={historySeries,renderHistoryProgress,smoothHistory};
})(typeof window!=='undefined'?window:globalThis);
