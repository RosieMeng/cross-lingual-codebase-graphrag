export default function EvidencePanel({ evidence = [], citationAudit, graphTraces = [] }) {
  return (
    <>
      {citationAudit?.status === 'unsupported' && (
        <p className="mt-3 text-xs text-amber-300">Some citations do not match the retrieved evidence: {(citationAudit.unsupported || []).join(', ')}</p>
      )}
      {citationAudit?.status === 'not_detected' && (
        <p className="mt-3 text-xs text-slate-400">No source ID citations detected in the answer.</p>
      )}
      {citationAudit?.status === 'matched' && (
        <p className="mt-3 text-xs text-slate-400">Citation IDs match retrieved evidence; factual support has not been verified.</p>
      )}
      {evidence.length > 0 && (
        <div className="mt-3 pt-3 border-t border-slate-700/30">
          <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">Retrieved evidence</h4>
          <div className="flex flex-wrap gap-1.5">
            {evidence.map(src => (
              <span key={src.id} className="text-[10px] px-2 py-0.5 bg-indigo-500/5 border border-indigo-500/10 text-indigo-400 rounded-md font-mono">
                [{src.citation_id}] {src.file}:{src.line_start}{src.line_end && src.line_end !== src.line_start ? `–${src.line_end}` : ''}{src.symbol ? ` · ${src.symbol}` : ''}
              </span>
            ))}
          </div>
        </div>
      )}
      {graphTraces.filter(graph => graph?.lookup_status && graph.lookup_status !== 'not_applicable').map((graph, index) => (
        <details key={index} className="mt-3 text-xs text-slate-400">
          <summary>Symbol and call-site information</summary>
          {graph.root ? (
            <p>Grounded {graph.root.qualified}: {graph.root.file}:{graph.root.line_start}–{graph.root.line_end}</p>
          ) : (
            <p>Exact symbol lookup: {graph.lookup_status}. No unique target was established.</p>
          )}
          {(graph.relationships || []).map((edge, i) => (
            <p key={`r-${i}`}>Static reference (runtime target not verified): {edge.caller} → {edge.callee} at {edge.caller_file}:{edge.call_line}{edge.caller_citation_id ? ` [${edge.caller_citation_id}]` : ' (source not retrieved)'}</p>
          ))}
          {(graph.unresolved || []).map((call, i) => (
            <p key={`u-${i}`}>Unresolved call; target not established: {call.expression} at {call.file}:{call.line}{call.caller_citation_id ? ` [${call.caller_citation_id}]` : ' (source not retrieved)'}</p>
          ))}
          {(graph.relationships_total > (graph.relationships || []).length || graph.unresolved_total > (graph.unresolved || []).length) && <p>Additional graph entries were omitted.</p>}
        </details>
      ))}
    </>
  );
}
