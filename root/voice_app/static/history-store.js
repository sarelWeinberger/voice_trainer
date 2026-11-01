/* Per-account cache: server restarts and empty cloud responses do not erase it. */
(function(root){
    'use strict';
    const LIMIT=10;
    function normalize(entry){
        if(!entry || typeof entry!=='object' || Array.isArray(entry))return null;
        const time=entry.timestamp || entry.date || entry.created_at;
        const stamp=Date.parse(time);
        if(!Number.isFinite(stamp))return null;
        const iso=new Date(stamp).toISOString();
        return {...entry,timestamp:iso,date:iso};
    }
    function merge(local,cloud=[]){
        const found=new Map();
        for(const source of [local,cloud]){
            for(const value of Array.isArray(source)?source:[]){
                const entry=normalize(value);if(!entry)continue;
                const identity=entry.session_id?'session:'+entry.session_id:'date:'+entry.timestamp;
                const old=found.get(identity);
                const rank=packet=>packet?.neckAnalysis?.status==='complete'?2:packet?.neckAnalysis?1:0;
                if(old){
                    const joined={...old,...entry};
                    if(rank(old.neck_effort)>rank(entry.neck_effort))joined.neck_effort=old.neck_effort;
                    // Cloud timestamps belong to the saved result; keep a stable chronological identity.
                    found.set(identity,joined);
                }else found.set(identity,entry);
            }
        }
        return [...found.values()].sort((a,b)=>Date.parse(b.timestamp)-Date.parse(a.timestamp)).slice(0,LIMIT);
    }
    function read(key,storage=root.localStorage){
        try{return merge(JSON.parse(storage.getItem(key)||'[]'));}catch(_){return [];}
    }
    function save(key,entries,storage=root.localStorage){
        const history=merge(entries);
        try{storage.setItem(key,JSON.stringify(history));}catch(error){
            if(root.console)root.console.warn('History cache could not be saved',error.name);
        }
        return history;
    }
    root.VoiceHistory={LIMIT,merge,read,save};
    if(typeof module!=='undefined')module.exports=root.VoiceHistory;
})(typeof window!=='undefined'?window:globalThis);
