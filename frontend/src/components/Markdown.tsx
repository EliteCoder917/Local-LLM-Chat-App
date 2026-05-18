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
            const raw = String(children ?? '').replace(/\n$/, '');
            if (lang === 'mermaid') {
              return <MermaidBlock source={raw} />;
            }
            return <CodeBlock code={raw} language={lang} className={className} />;
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
