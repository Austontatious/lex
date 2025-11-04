// tailwind.config.js
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        friday: {
          cyan: "#00FFFF",
          black: "#111111",
          white: "#FFFFFF",
        },
        background: "#111111",
        primary: "#00FFFF",
        secondary: "#00CCCC",
        surface: "#1a1a1a",
        muted: "#666666",
      },
      fontFamily: {
      friday: ['"Barlow Condensed"', "sans-serif"],

      },
      letterSpacing: {
        wide: "0.08em",
      },
      boxShadow: {
        cyanGlow: "0px 0px 12px #00FFFF",
        whiteGlow: "0px 0px 18px #FFFFFF",
        blackGlow: "0px 0px 10px #111111",
        blackText: "2px 2px 6px #111111",
        cyanText: "2px 2px 8px #00FFFF",
        whiteText: "2px 2px 12px #FFFFFF",
      },
      animation: {
        fridayPulse: "fridayPulse 5s ease-in-out infinite",
      },
      keyframes: {
        fridayPulse: {
          "0%": {
            boxShadow: "0 0 12px #00FFFF, 0 0 24px #00FFFF55",
          },
          "50%": {
            boxShadow: "0 0 16px #00FFFF, 0 0 32px #00FFFF99",
          },
          "100%": {
            boxShadow: "0 0 12px #00FFFF, 0 0 24px #00FFFF55",
          },
        },
      },
    },
  },
  plugins: [],
};

