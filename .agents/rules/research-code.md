---
trigger: always_on
---

Based on the principles of clean coding for AI agents and general project guidelines, here is the ultimate rule book for coding style to ensure your MICA project remains robust, maintainable, and aligned with Clean Code principles.
Writing Clean Functions

Functions must adhere to strict single-responsibility boundaries. Limit functions to a maximum of three arguments; if more are required, encapsulate them within a data structure (e.g., Python dataclasses). Avoid using boolean flag arguments, as they indicate a function does more than one thing; split these into separate functions instead. Furthermore, eliminate all output arguments by modifying arguments as side effects—return new values instead. Ensure you aggressively delete dead or unused functions rather than keeping them "just in case," relying on Git for history retention.
Clear and Concise and Precise word usuage  Comments

Code should be largely self-documenting; write comments only to explain "why" rather than "what". Do not include metadata like authors, change history, or dates in comments, as this information belongs in version control. Promptly delete obsolete comments to prevent misdirection, and never commit commented-out code. If a comment is required to explain complex logic, consider whether the code can be refactored for clarity first.
General Clean Architecture

Adhere strictly to the DRY (Don't Repeat Yourself) principle, ensuring every piece of logic has a single authoritative representation. Avoid clever or obscured logic; code intent must be immediately obvious (e.g., naming variables clearly instead of relying on bitwise tricks for coordinates). Replace "magic numbers" with clearly named constants to improve readability. Prefer polymorphism or class-based structures over long if/elif/else chains to ensure the codebase remains scalable under the Open/Closed Principle.
The Boy-Scout Principle for Iteration

Apply the "Boy Scout" rule to all coding workflows: always leave the code cleaner than you found it. When adding new logic for MICA's latent intent recognition, simultaneously clean up adjacent duplication or obscure variables.
