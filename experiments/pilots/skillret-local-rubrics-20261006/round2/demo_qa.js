(function(){
  const failures=[];let visited=0;
  try{
    data.panels.forEach((panel,pi)=>{
      byId('cohort').value=String(pi);byId('cohort').dispatchEvent(new Event('change'));
      panel.rows.forEach((row,qi)=>{
        byId('query').value=String(qi);byId('query').dispatchEvent(new Event('change'));visited++;
        if(byId('dense').querySelectorAll('li').length!==10)failures.push('dense count '+row.id);
        if(byId('fusion').querySelectorAll('li').length!==10)failures.push('served count '+row.id);
        if(byId('task').textContent!==row.query)failures.push('task text '+row.id);
        if(row.status==='ok'&&byId('jev').querySelectorAll('li').length!==10)failures.push('rerank count '+row.id);
      });
    });
    byId('cohort').value='2';byId('cohort').dispatchEvent(new Event('change'));
    byId('failure').click();if(!byId('summary').textContent.includes('Fallback active'))failures.push('failure toggle');
    if(byId('jev').querySelectorAll('li').length!==0)failures.push('failure rerank not empty');
    byId('failure').click();if(byId('summary').textContent.includes('Fallback active'))failures.push('restore toggle');
    if(document.documentElement.scrollWidth>window.innerWidth+2)failures.push('horizontal overflow');
  }catch(error){failures.push('runtime error');}
  document.body.setAttribute('data-qa',JSON.stringify({visited,failures}));
})();
