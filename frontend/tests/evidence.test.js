import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let vite;
let EvidencePanel;
before(async () => {
  vite = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
  EvidencePanel = (await vite.ssrLoadModule('/src/components/EvidencePanel.jsx')).default;
});
after(async () => { if (vite) await vite.close(); });

test('renders source IDs with the exact file, symbol and source range', () => {
  const html = renderToStaticMarkup(React.createElement(EvidencePanel, {
    evidence: [{ id: 'abc', citation_id: 'S1', file: 'micrograd/engine.py', symbol: 'backward', line_start: 54, line_end: 70 }],
    citationAudit: { status: 'matched', unsupported: [] },
  }));
  assert.match(html, /\[S1\] micrograd\/engine\.py:54–70 · backward/);
  assert.match(html, /factual support has not been verified/);
});

test('warns about nonexistent source IDs and escapes evidence text', () => {
  const html = renderToStaticMarkup(React.createElement(EvidencePanel, {
    evidence: [{ id: 'a', citation_id: 'S1', file: '<unsafe>.py', line_start: 1 }],
    citationAudit: { status: 'unsupported', unsupported: ['[S7]'] },
  }));
  assert.match(html, /\[S7\]/);
  assert.match(html, /&lt;unsafe&gt;\.py/);
  assert.doesNotMatch(html, /<unsafe>/);
});

test('reports missing citations without inventing evidence', () => {
  const html = renderToStaticMarkup(React.createElement(EvidencePanel, {
    citationAudit: { status: 'not_detected', unsupported: [] },
  }));
  assert.match(html, /No source ID citations/);
  assert.doesNotMatch(html, /Retrieved evidence/);
});
