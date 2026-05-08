export default [
  {
    ignores: ["node_modules/**", "dist/**", "out/**", ".cache_ggshield/**", ".fallow/**"],
  },
  {
    files: ["src/**/*.{js,ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
    },
    rules: {
      "no-unused-vars": "off",
      "no-undef": "off",
    },
  },
];
