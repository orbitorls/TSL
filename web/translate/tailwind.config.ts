import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          DEFAULT: "#0f4f4a",
          light: "#13a08f",
          ghost: "#e8f4f1",
        },
        page: "#f6f4ef",
        panel: "#ffffff",
        border: "#d8d0bf",
        subtle: "#5f6660",
        text: "#23312f",
      },
      fontFamily: {
        thai: ['"Noto Sans Thai"', "Sarabun", "Prompt", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
