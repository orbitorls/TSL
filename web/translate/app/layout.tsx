import type { Metadata } from "next";
import { Noto_Sans_Thai } from "next/font/google";
import "./globals.css";

const noto = Noto_Sans_Thai({
  subsets: ["thai", "latin"],
  weight: ["400", "600", "700"],
  variable: "--font-noto-thai",
});

export const metadata: Metadata = {
  title: "TSL แปลภาษามือ",
  description: "ระบบแปลภาษามือไทยเป็นข้อความแบบเรียลไทม์",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="th" className={noto.variable}>
      <body className={noto.className} suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
