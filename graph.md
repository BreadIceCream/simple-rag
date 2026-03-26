---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	init_graph_state(init_graph_state)
	compact_tool_messages(compact_tool_messages)
	summarize_conversation(summarize_conversation)
	decide_retrieve_or_respond(decide_retrieve_or_respond)
	retrieve(retrieve)
	generate_answer(generate_answer)
	rewrite_question(rewrite_question)
	check_usefulness_node(check_usefulness_node)
	__end__([<p>__end__</p>]):::last
	__start__ --> init_graph_state;
	check_usefulness_node -.-> __end__;
	check_usefulness_node -.-> rewrite_question;
	compact_tool_messages --> summarize_conversation;
	decide_retrieve_or_respond -.-> __end__;
	decide_retrieve_or_respond -. &nbsp;tools&nbsp; .-> retrieve;
	generate_answer -. &nbsp;check_usefulness&nbsp; .-> check_usefulness_node;
	init_graph_state --> compact_tool_messages;
	retrieve -.-> decide_retrieve_or_respond;
	retrieve -.-> generate_answer;
	retrieve -.-> rewrite_question;
	rewrite_question --> decide_retrieve_or_respond;
	summarize_conversation --> decide_retrieve_or_respond;
	generate_answer -.-> generate_answer;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
