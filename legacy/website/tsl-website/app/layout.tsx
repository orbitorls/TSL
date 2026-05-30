import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'TSL-51 | Thai Sign Language Translator',
  description: 'Real-time Thai Sign Language translation using AI',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="th">
      <head>
        <link
          href="https://api.fontshare.com/v2/css?f[]=clash-display@400,500,600,700&f[]=satoshi@400,500,700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen bg-surface-50">{children}</body>
    </html>
  )
}