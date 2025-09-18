import ChatBot from "../components/ChatBot";
import { WorkflowTrigger } from "@llamaindex/ui";
import { APP_TITLE } from "../libs/config";

export default function Home() {
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

        <div>
          <div className="flex mb-4">
            <WorkflowTrigger
              workflowName="upload"
              inputFields={[{
                key: "index_name",
                label: "Index Name",
                placeholder: "e.g. document_qa_index",
                required: true,
              }]}
              customWorkflowInput={(files, fieldValues) => {
                return {
                  file_id: files[0].fileId,
                  index_name: fieldValues.index_name,
                };
              }}
            />
          </div>
          <div>
            <div className="h-[700px]">
              <ChatBot />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}