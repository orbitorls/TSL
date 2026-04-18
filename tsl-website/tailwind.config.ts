import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f2fcfb',
          100: '#dcfaf5',
          200: '#b9f3ea',
          300: '#86e5d8',
          400: '#4dd3c0',
          500: '#2ac4ae',
          600: '#1fa494',
          700: '#1a857a',
          800: '#1a6a63',
          900: '#175652',
        },
        surface: {
          50: '#fafcfc',
          100: '#f5f9f8',
          200: '#e9f2f0',
          300: '#d5e6e2',
          400: '#b5d1cb',
          500: '#94bab2',
          600: '#7aa39b',
        }
      },
      fontFamily: {
        display: ['Clash Display', 'system-ui', 'sans-serif'],
        body: ['Satoshi', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.6s ease-out',
        'pulse-soft': 'pulseSoft 2s infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
      },
    },
  },
  plugins: [],
}
export default config