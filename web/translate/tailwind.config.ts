import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          DEFAULT: "#0f4f4a",
          dark: "#083a36",
          light: "#13a08f",
          soft: "#dcefe9",
          ghost: "#eef7f4",
        },
        page: "#f6f4ef",
        panel: "#ffffff",
        border: "#d8d0bf",
        line: "#ebe4d6",
        subtle: "#5f6660",
        muted: "#7b817b",
        text: "#23312f",
        ink: "#172522",
        success: "#16755f",
        warning: "#9a5b13",
        danger: "#b42318",
      },
      boxShadow: {
        shell: "0 24px 70px rgba(35, 49, 47, 0.10)",
        panel: "0 14px 34px rgba(35, 49, 47, 0.08)",
        chip: "inset 0 0 0 1px rgba(15, 79, 74, 0.10)",
      },
      fontFamily: {
        thai: ['"Noto Sans Thai"', "Sarabun", "Prompt", "system-ui", "sans-serif"],
      },
      letterSpacing: {
        operator: "0.08em",
      },
    },
  },
  plugins: [],
};

export default config;
