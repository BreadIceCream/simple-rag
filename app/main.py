import asyncio

from langgraph.graph import StateGraph, START, END

from app.dependencies import init_rag_application
from app.models.state import OverallState
from app.core.graph import generate_query_or_respond, tool_node, generate_answer, should_use_tool


if __name__ == "__main__":
    init_result = asyncio.run(init_rag_application())
    llm = init_result["llm"]
    embeddings = init_result["embeddings"]
    vector_store = init_result["vector_store"]
    collection = init_result["collection"]
    client = init_result["client"]
    text_splitter = init_result["text_splitter"]
    docs_ids = init_result["docs_ids"]
    hybrid_retriever = init_result["hybrid_retriever"]
    compression_retriever = init_result["compression_retriever"]

    # 创建Graph
    workflow = StateGraph(OverallState)

    workflow.add_node(generate_query_or_respond)
    workflow.add_node(tool_node)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges("generate_query_or_respond", should_use_tool)
    workflow.add_edge("tool_node", "generate_answer")
    workflow.add_edge("generate_answer", END)

    agent = workflow.compile()

    # 运行
    command = "retrieve"
    while True:
        print(f"====================================================================")
        print(
            f"Using collection '{collection.name}'. Current mode '{command}'. Enter '/{['retrieve', 'direct'][command == 'retrieve']}' to switch to {['retrieve', 'direct'][command == 'retrieve']} mode.")
        user_input = input(f"Ask your questions (input 'exit' to stop)：")
        if user_input == "exit":
            if docs_ids:
                delete = input(
                    "Do you want to delete the documents added in this session? If not, they will persist in the current folder. (y/n) ")
                while delete not in ["y", "n"]:
                    delete = input("Invalid input. Please enter 'y' or 'n': ")
                if delete == "y":
                    collection.delete(ids=docs_ids)
                    if collection.count() == 0:
                        print(f"No documents.Delete collection {collection.name}.")
                        client.delete_collection(collection.name)
                    print("Documents deleted.")
                else:
                    print("Documents will persist in the current folder.")
            print("Bye!")
            break
        elif user_input.startswith("/"):
            if user_input not in ["/retrieve", "/direct"]:
                print("Invalid command. Please enter '/retrieve' or '/direct'.")
                continue
            command = user_input[1:]
            print(f"Switched to {command} mode.\n")
            continue
        state = {"command": command, "user_question": user_input, "retrieved_docs": []}
        for chunk in agent.stream(
                input=state,
        ):
            for node, update in chunk.items():
                print("Update from node", node)
                update["messages"][-1].pretty_print()
                print("\n\n")
        if command == "retrieve" and state["retrieved_docs"]:
            print(f"Documents\n: {state['retrieved_docs']}")
