import type { Metadata, Viewport } from "next";
import Header from "@/components/Header";
import { ProgressProvider } from "@/components/ProgressProvider";
import { THEME_SCRIPT } from "@/components/ThemeToggle";
import { COURSE } from "@/lib/modules";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: `${COURSE.title} · ${COURSE.program}`, template: `%s · ${COURSE.title}` },
  description:
    "An 80-day video course in applied causal inference and decision science: data engineering, DAGs, experiments, quasi-experiments, causal machine learning and decision tools.",
  authors: [{ name: COURSE.author }],
  creator: COURSE.author,
};

export const viewport: Viewport = { themeColor: "#020617" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>
        <ProgressProvider>
          <Header />
          <main className="mx-auto max-w-[1440px] px-4 py-6 sm:px-6 sm:py-8">{children}</main>
          <footer className="mt-10 border-t border-slate-800">
            <div className="mx-auto flex max-w-[1440px] flex-col justify-between gap-2 px-4 py-6 text-xs text-slate-500 sm:flex-row sm:px-6">
              <span>
                {COURSE.program}: {COURSE.title} · Created by {COURSE.author}
              </span>
              <span>Case study: NorthMart, a simulated retailer with known causal effects · Progress is saved in your browser only.</span>
            </div>
          </footer>
        </ProgressProvider>
      </body>
    </html>
  );
}
