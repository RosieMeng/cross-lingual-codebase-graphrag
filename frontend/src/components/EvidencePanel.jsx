export default function EvidencePanel({ evidence = [], citationAudit }) {
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
    </>
  );
}
