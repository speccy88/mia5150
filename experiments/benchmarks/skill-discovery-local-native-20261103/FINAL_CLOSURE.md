# Final closure, 3 November 2026

The initial local-native-v2 series remains unchanged: 14 configurations measured,
four blocked at 32K/f16. The separate profiles-v3 supplement adds Qwen8 and Mistral
measurements at adjusted contexts with f16, plus paired Q8 KV tests at N=500.
There are 16 distinct measured local models across the two series, 1,320 measured
local responses, plus 360 measured OAuth responses. The 200 v3 responses are not
silently merged with the initial protocol or its 10% estimate margin.

Llama 3.2 and Gemma 3 remain not executed; further access work was declined by the
user. No gated acceptance or substitute was attempted. The position bias and
CPU_Mapped input-weight participation remain explicit. No new inference was run
while preparing this public integration. Original ZIPs and the earlier provisional
integration snapshot are retained unchanged privately.
