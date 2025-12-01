"""
Next.js static code analyzer.

Provides static analysis for Next.js applications with support for:
- App Router (app/) - Next.js 13+
- Pages Router (pages/) - Traditional Next.js
- Server Components and Server Actions
- File-system based routing
- Data fetching methods (getServerSideProps, getStaticProps)
- Next.js specific hooks (useRouter, useSearchParams, usePathname, useParams)
"""

from qontinui.extraction.static.nextjs.analyzer import NextJSStaticAnalyzer

__all__ = ["NextJSStaticAnalyzer"]
