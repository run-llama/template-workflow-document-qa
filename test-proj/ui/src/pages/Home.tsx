import ChatBot from "../components/ChatBot";
import {
  useWorkflowHandlerList,
  WorkflowProgressBar,
  WorkflowTrigger,
} from "@llamaindex/ui";
import { APP_TITLE, INDEX_NAME } from "../libs/config";
import { useChatHistory } from "@/libs/useChatHistory";
import Sidebar from "@/components/Sidebar";
import { Loader } from "lucide-react";

export default function Home() {
  const chatHistory = useChatHistory();
  const handlers = useWorkflowHandlerList("upload");
  const activeHandlers = handlers.handlers.filter(
    (h) => h.status === "running" && h.workflowName === "upload",
  );
  const anyActiveHandlers = activeHandlers.length > 0;
  console.log("activeHandlers", activeHandlers);
  console.log("anyActiveHandlers", anyActiveHandlers);
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            {APP_TITLE}
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Upload documents and ask questions about them
          </p>
        </header>

        <div className="flex gap-4 w-full items-stretch">
          <Sidebar chatHistory={chatHistory} />
          <div className="w-full">
            <div className="flex mb-4">
              <WorkflowTrigger
                workflowName="upload"
                customWorkflowInput={(files, fieldValues) => {
                  return {
                    file_id: files[0].fileId,
                    index_name: INDEX_NAME,
                  };
                }}
              />
              {anyActiveHandlers && (
                <div className="ml-3 flex items-center justify-center">
                  <Loader className="w-4 h-4 animate-spin" />
                </div>
              )}
            </div>
            <div>
              <div className="h-[700px]">
                {!chatHistory.loading && (
                  <ChatBot
                    key={`${chatHistory.chatCounter}-${chatHistory.selectedChatId || "new"}`}
                    handlerId={chatHistory.selectedChatId ?? undefined}
                    onHandlerCreated={(handler) => {
                      chatHistory.addChat(handler);
                      chatHistory.setSelectedChatId(handler);
                    }}
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
