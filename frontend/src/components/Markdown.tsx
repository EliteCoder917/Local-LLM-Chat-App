import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import 'highlight.js/styles/github-dark.css';
import 'katex/dist/katex.min.css';
import CodeBlock from './CodeBlock';
import MermaidBlock from './MermaidBlock';

// rehypeHighlight transforms fenced code blocks into nested <span> trees so
// highlight.js classes can colour individual tokens. That means by the time
// our `code` component runs, `children` is an ARRAY of React elements, not a
// flat string. `String(array)` on that produces `[object Object],…` garbage —
// the exact symptom in the chat bubble. Recurse the tree pulling out only
// text nodes so we get back the raw source the user typed.
function nodeToText(node: React.ReactNode): string {
  if (node == null || typeof node === 'boolean') return '';
  if (typeof node === 'string') return node;
  if (typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(nodeToText).join('');
  if (typeof node === 'object' && 'props' in (node as object)) {
    return nodeToText((node as React.ReactElement<{ children?: React.ReactNode }>).props.children);
  }
  return '';
}

export default function Markdown({ text }: { text: string }) {
  return (
    <div className="prose prose-invert prose-sm max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeHighlight, rehypeKatex]}
        components={{
          // Replace fenced code blocks with our richer CodeBlock (copy +
          // language badge + optional Apply-to-file in Code tab). Inline
          // code (`like this`) falls through unchanged. `mermaid` blocks
          // get rendered as diagrams.
          //
          // react-markdown v9 dropped the `inline` prop. Detect fenced
          // blocks instead by the language-* class the parser attaches —
          // inline backticks never carry one. Without this, inline code
          // would render as a <div>/<pre> inside a <p>, which is invalid
          // DOM nesting and React shouts about it.
          code({ className, children, ...rest }: any) {
            const lang = className?.match(/language-(\S+)/)?.[1] ?? '';
            if (!lang) {
              return (
                <code
                  className="px-1 py-0.5 rounded bg-neutral-800/60 text-neutral-200 text-[12.5px] mono"
                  {...rest}
                >
                  {children}
                </code>
              );
            }
            const raw = nodeToText(children).replace(/\n$/, '');
            if (lang === 'mermaid') {
              return <MermaidBlock source={raw} />;
            }
            // Pass the already-highlighted children too so CodeBlock can keep
            // syntax colouring (those <span class="hljs-*"> nodes are what
            // makes the styles in github-dark.css actually fire). `raw` still
            // backs Copy / Apply so they get clean text.
            return (
              <CodeBlock
                code={raw}
                language={lang}
                className={className}
                highlighted={children}
              />
            );
          },
          // Wrap react-markdown's default <pre> as a transparent passthrough
          // since CodeBlock provides its own <pre>.
          pre({ children }) {
            return <>{children}</>;
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
