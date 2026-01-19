# Structured Debugging Skill

## Overview

This skill enforces structured reasoning during debugging tasks. You MUST follow this protocol when debugging.

## Protocol

### Before Running ANY Test

1. **Create/Update state.json** with:
   - At least one hypothesis in `hypotheses`
   - Current probe with `intent` and `prediction`

2. **Format for current_probe**:
```json
{
  "current_probe": {
    "intent": "Describe what you're testing and why",
    "prediction": {
      "if_pass": "What it means if the test passes",
      "if_fail": "What it means if the test fails"
    },
    "test_command": "The exact test command"
  }
}
```

### After Each Test

1. **Update history** with:
   - `result`: "pass" or "fail"
   - `update`: What you learned and how hypotheses changed

2. **Update hypotheses**:
   - Mark confirmed/eliminated based on evidence
   - Add new hypotheses if discovered

### Using Dataflow Tools

When investigating root causes, use MCP tools:

- `trace_symbol`: Find where a variable/function is defined and used
- `trace_callchain`: Understand call hierarchy

Record MCP results in `dataflow_chain` field.

## Example Workflow

1. Read bug report → form initial hypothesis
2. Create state.json with hypothesis
3. Fill current_probe with intent/prediction
4. Run test (gate will verify state.json)
5. Record result and update
6. Refine hypothesis or fix

## Anti-Patterns to Avoid

❌ Running tests without stating intent
❌ "Try and see" without prediction
❌ Patching symptoms without tracing root cause
❌ Empty try/except blocks
❌ Skipping tests instead of fixing them
