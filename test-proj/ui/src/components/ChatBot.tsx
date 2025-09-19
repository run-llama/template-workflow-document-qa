// This is a temporary chatbot component that is used to test the chatbot functionality.
// LlamaIndex will replace it with better chatbot component.
import { useState, useRef, useEffect, FormEvent, KeyboardEvent } from "react";
import {
  Send,
  Loader2,
  Bot,
  User,
  MessageSquare,
  Trash2,
  RefreshCw,
} from "lucide-react";
import {
  Button,
  Input,
  ScrollArea,
  Card,
  CardContent,
  cn,
  useWorkflowRun,
  useWorkflowHandler,
} from "@llamaindex/ui";
import { AGENT_NAME } from "../libs/config";
import { toHumanResponseRawEvent } from "@/libs/utils";

type Role = "user" | "assistant";
interface Message {
  id: string;
  role: Role;
  content: string;
  timestamp: Date;
  error?: boolean;
}
export default function ChatBot() {
  const { runWorkflow } = useWorkflowRun();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [handlerId, setHandlerId] = useState<string | null>(null);
  const lastProcessedEventIndexRef = useRef<number>(0);
  const [canSend, setCanSend] = useState<boolean>(false);
  const streamingMessageIndexRef = useRef<number | null>(null);

  // Deployment + auth setup
  const deployment = AGENT_NAME || "document-qa";
  const platformToken = (import.meta as any).env?.VITE_LLAMA_CLOUD_API_KEY as
    | string
    | undefined;
  const projectId = (import.meta as any).env?.VITE_LLAMA_DEPLOY_PROJECT_ID as
    | string
    | undefined;
  const defaultIndexName =
    (import.meta as any).env?.VITE_DEFAULT_INDEX_NAME || "document_qa_index";
  const sessionIdRef = useRef<string>(
    `chat-${Math.random().toString(36).slice(2)}-${Date.now()}`,
  );

  // UI text defaults
  const title = "AI Document Assistant";
  const placeholder = "Ask me anything about your documents...";
  const welcomeMessage =
    "Welcome! 👋 Upload a document with the control above, then ask questions here.";

  // Helper functions for message management
  const appendMessage = (role: Role, msg: string): void => {
    setMessages((prev) => {
      const id = `${role}-stream-${Date.now()}`;
      const idx = prev.length;
      streamingMessageIndexRef.current = idx;
      return [
        ...prev,
        {
          id,
          role,
          content: msg,
          timestamp: new Date(),
        },
      ];
    });
  };

  const updateMessage = (index: number, message: string) => {
    setMessages((prev) => {
      if (index < 0 || index >= prev.length) return prev;
      const copy = [...prev];
      const existing = copy[index];
      copy[index] = { ...existing, content: message };
      return copy;
    });
  };

  // Initialize with welcome message
  useEffect(() => {
    if (messages.length === 0) {
      const welcomeMsg: Message = {
        id: "welcome",
        role: "assistant",
        content: welcomeMessage,
        timestamp: new Date(),
      };
      setMessages([welcomeMsg]);
    }
  }, []);

  // Create chat task on init
  useEffect(() => {
    (async () => {
      if (!handlerId) {
        const handler = await runWorkflow("chat", {
          index_name: defaultIndexName,
          session_id: sessionIdRef.current,
        });
        setHandlerId(handler.handler_id);
      }
    })();
  }, []);

  // Subscribe to task/events using hook (auto stream when handler exists)
  const { events } = useWorkflowHandler(handlerId ?? "", Boolean(handlerId));

  // Process streamed events into messages
  useEffect(() => {
    if (!events || events.length === 0) return;
    let startIdx = lastProcessedEventIndexRef.current;
    if (startIdx < 0) startIdx = 0;
    if (startIdx >= events.length) return;

    for (let i = startIdx; i < events.length; i++) {
      const ev: any = events[i];
      const type = ev?.type as string | undefined;
      const rawData = ev?.data as any;
      if (!type) continue;
      const data = (rawData && (rawData._data ?? rawData)) as any;

      if (type.includes("ChatDeltaEvent")) {
        const delta: string = data?.delta ?? "";
        if (!delta) continue;
        if (streamingMessageIndexRef.current === null) {
          appendMessage("assistant", delta);
        } else {
          const idx = streamingMessageIndexRef.current;
          const current = messages[idx!]?.content ?? "";
          if (current === "Thinking...") {
            updateMessage(idx!, delta);
          } else {
            updateMessage(idx!, current + delta);
          }
        }
      } else if (type.includes("ChatResponseEvent")) {
        // finalize current stream
        streamingMessageIndexRef.current = null;
      } else if (type.includes("InputRequiredEvent")) {
        // ready for next user input; enable send
        setCanSend(true);
        setIsLoading(false);
        inputRef.current?.focus();
      } else if (type.includes("StopEvent")) {
        // finished; no summary bubble needed (chat response already streamed)
      }
    }
    lastProcessedEventIndexRef.current = events.length;
  }, [events, messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // No manual SSE cleanup needed

  const getCommonHeaders = () => ({
    ...(platformToken ? { authorization: `Bearer ${platformToken}` } : {}),
    ...(projectId ? { "Project-Id": projectId } : {}),
  });

  const startChatIfNeeded = async (): Promise<string> => {
    if (handlerId) return handlerId;
    const handler = await runWorkflow("chat", {
      index_name: defaultIndexName,
      session_id: sessionIdRef.current,
    });
    setHandlerId(handler.handler_id);
    return handler.handler_id;
  };

  // Removed manual SSE ensureEventStream; hook handles streaming

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    const trimmedInput = input.trim();
    if (!trimmedInput || isLoading || !canSend) return;

    // Add user message
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmedInput,
      timestamp: new Date(),
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput("");
    setIsLoading(true);
    setCanSend(false);

    // Immediately create an assistant placeholder to avoid visual gap before deltas
    if (streamingMessageIndexRef.current === null) {
      appendMessage("assistant", "Thinking...");
    }

    try {
      // Ensure chat handler exists (created on init)
      const hid = await startChatIfNeeded();

      // Send user input as HumanResponseEvent
      const postRes = await fetch(`/deployments/${deployment}/events/${hid}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getCommonHeaders(),
        },
        body: JSON.stringify({
          event: JSON.stringify(toHumanResponseRawEvent(trimmedInput)),
        }),
      });
      if (!postRes.ok) {
        throw new Error(
          `Failed to send message: ${postRes.status} ${postRes.statusText}`,
        );
      }

      // The assistant reply will be streamed by useWorkflowTask and appended incrementally
    } catch (err) {
      console.error("Chat error:", err);

      // Add error message
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: "assistant",
        content: `Sorry, I encountered an error: ${err instanceof Error ? err.message : "Unknown error"}. Please try again.`,
        timestamp: new Date(),
        error: true,
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      // Focus back on input
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    // Submit on Enter (without Shift)
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  const clearChat = () => {
    setMessages([
      {
        id: "welcome",
        role: "assistant" as const,
        content: welcomeMessage,
        timestamp: new Date(),
      },
    ]);
    setInput("");
    inputRef.current?.focus();
  };

  const retryLastMessage = () => {
    const lastUserMessage = messages.filter((m) => m.role === "user").pop();
    if (lastUserMessage) {
      // Remove the last assistant message if it was an error
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.role === "assistant" && lastMessage.error) {
        setMessages((prev) => prev.slice(0, -1));
      }
      setInput(lastUserMessage.content);
      inputRef.current?.focus();
    }
  };

  return (
    <div
      className={cn(
        "flex flex-col h-full bg-white dark:bg-gray-800 rounded-lg shadow-lg",
      )}
    >
      {/* Header */}
      <div className="px-4 py-3 border-b dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <h3 className="font-semibold text-gray-900 dark:text-white">
              {title}
            </h3>
            {isLoading && (
              <span className="text-xs text-gray-500 dark:text-gray-400">
                Thinking...
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {messages.some((m) => m.error) && (
              <button
                onClick={retryLastMessage}
                className="p-1.5 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                title="Retry last message"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            )}
            {messages.length > 0 && (
              <button
                onClick={clearChat}
                className="p-1.5 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                title="Clear chat"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full min-h-[300px]">
            <div className="text-center">
              <Bot className="w-12 h-12 text-gray-400 dark:text-gray-600 mx-auto mb-3" />
              <p className="text-gray-500 dark:text-gray-400 mb-2">
                No messages yet
              </p>
              <p className="text-sm text-gray-400 dark:text-gray-500">
                Start a conversation!
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  "flex gap-3",
                  message.role === "user" ? "justify-end" : "justify-start",
                )}
              >
                {message.role !== "user" && (
                  <div
                    className={cn(
                      "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
                      message.error
                        ? "bg-red-100 dark:bg-red-900"
                        : "bg-blue-100 dark:bg-blue-900",
                    )}
                  >
                    <Bot
                      className={cn(
                        "w-5 h-5",
                        message.error
                          ? "text-red-600 dark:text-red-400"
                          : "text-blue-600 dark:text-blue-400",
                      )}
                    />
                  </div>
                )}
                <div
                  className={cn(
                    "max-w-[70%]",
                    message.role === "user" ? "order-1" : "order-2",
                  )}
                >
                  <Card
                    className={cn(
                      "py-0",
                      message.role === "user"
                        ? "bg-blue-600 text-white border-blue-600"
                        : message.error
                          ? "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800"
                          : "bg-gray-50 dark:bg-gray-700",
                    )}
                  >
                    <CardContent className="p-3">
                      <p
                        className={cn(
                          "whitespace-pre-wrap text-sm",
                          message.error && "text-red-700 dark:text-red-400",
                        )}
                      >
                        {message.content}
                      </p>
                      <p
                        className={cn(
                          "text-xs mt-1 opacity-70",
                          message.role === "user"
                            ? "text-blue-100"
                            : message.error
                              ? "text-red-500 dark:text-red-400"
                              : "text-gray-500 dark:text-gray-400",
                        )}
                      >
                        {message.timestamp.toLocaleTimeString()}
                      </p>
                    </CardContent>
                  </Card>
                </div>
                {message.role === "user" && (
                  <div className="w-8 h-8 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center flex-shrink-0 order-2">
                    <User className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                  </div>
                )}
              </div>
            ))}

            {isLoading && (
              <div className="flex gap-3 justify-start">
                <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
                  <Bot className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                </div>
                <Card className="bg-gray-50 dark:bg-gray-700 py-0">
                  <CardContent className="p-3">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <span
                          className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                          style={{ animationDelay: "0ms" }}
                        ></span>
                        <span
                          className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                          style={{ animationDelay: "150ms" }}
                        ></span>
                        <span
                          className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                          style={{ animationDelay: "300ms" }}
                        ></span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </ScrollArea>

      {/* Input */}
      <div className="border-t dark:border-gray-700 p-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={isLoading}
            className="flex-1"
            autoFocus
          />
          <Button
            type="submit"
            disabled={!canSend || isLoading || !input.trim()}
            size="icon"
            title="Send message"
          >
            {!canSend || isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </Button>
        </form>
        <p className="text-xs text-gray-400 dark:text-gray-500 mt-2">
          Press Enter to send • Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
