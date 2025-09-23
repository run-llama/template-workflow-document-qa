import { useWorkflowHandler, useWorkflowRun } from "@llamaindex/ui";
import { useEffect, useState } from "react";
import { INDEX_NAME } from "./config";

/**
 * Creates a new chat conversation if no handlerId is provided
 */
export function useChatWorkflowHandler({
  handlerId,
  onHandlerCreated,
}: {
  handlerId?: string;
  onHandlerCreated?: (handlerId: string) => void;
}): ReturnType<typeof useWorkflowHandler> {
  const create = useWorkflowRun();
  const [thisHandlerId, setThisHandlerId] = useState<string | undefined>(
    handlerId
  );
  const workflowHandler = useWorkflowHandler(thisHandlerId ?? "");

  const createHandler = async () => {
    const handler = await create.runWorkflow("chat", {
      index_name: INDEX_NAME,
    });
    setThisHandlerId(handler.handler_id);
    onHandlerCreated?.(handler.handler_id);
  };
  useEffect(() => {
    if (!handlerId) {
      createHandler();
    }
  }, [handlerId]);

  return workflowHandler;
}
