import js from "@eslint/js";
import globals from "globals";
export default [js.configs.recommended,{files:["web/**/*.js"],languageOptions:{globals:globals.browser}},{files:["scripts/**/*.mjs"],languageOptions:{globals:globals.node}}];
