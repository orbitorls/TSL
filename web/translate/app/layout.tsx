import type { Metadata } from "next";
import { IBM_Plex_Mono, Noto_Sans_Thai } from "next/font/google";
import "./globals.css";

const noto = Noto_Sans_Thai({
  subsets: ["thai", "latin"],
  weight: ["400", "500", "600", "700", "800"],
  variable: "--font-noto-thai",
  display: "swap",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "TSL แปลภาษามือ",
  description: "ระบบแปลภาษามือไทยเป็นข้อความแบบเรียลไทม์",
};

// Blocking inline script — reads localStorage + system pref and sets .dark on <html>
// BEFORE any paint, preventing flash of wrong theme.
const noFlashScript = `(function(){try{var t=localStorage.getItem('tsl-theme');var m=window.matchMedia('(prefers-color-scheme: dark)').matches;if(t==='dark'||(t===null&&m)){document.documentElement.classList.add('dark');}}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // suppressHydrationWarning because the inline script mutates className before hydration
    <html lang="th" className={`${noto.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        {/* Must be first in <head> to run before any paint */}
        {/* eslint-disable-next-line @next/next/no-sync-scripts */}
        <script dangerouslySetInnerHTML={{ __html: noFlashScript }} />
      </head>
      <body className="font-thai" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
