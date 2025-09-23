// This is a temporary chatbot component that is used to test the chatbot functionality.
// LlamaIndex will replace it with better chatbot component.
import { useChatbot } from "@/libs/useChatbot";
import {
  Button,
  Card,
  CardContent,
  cn,
  Input,
  ScrollArea,
} from "@llamaindex/ui";
import {
  Bot,
  Loader2,
  MessageSquare,
  RefreshCw,
  Send,
  Trash2,
  User,
} from "lucide-react";
import { FormEvent, KeyboardEvent, useEffect, useRef } from "react";

export default function ChatBot({
  handlerId,
  onHandlerCreated,
}: {
  handlerId?: string;
  onHandlerCreated?: (handlerId: string) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatbot = useChatbot({
    handlerId,
    onHandlerCreated,
    focusInput: () => {
      inputRef.current?.focus();
    },
  });

  // UI text defaults
  const title = "AI Document Assistant";
  const placeholder = "Ask me anything about your documents...";

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatbot.messages]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    await chatbot.submit();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    // Submit on Enter (without Shift)
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  return (
    <div
      className={cn(
        "flex flex-col h-full bg-white dark:bg-gray-800 rounded-lg shadow-lg"
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
            {chatbot.isLoading && (
              <span className="text-xs text-gray-500 dark:text-gray-400">
                Thinking...
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {chatbot.messages.some((m) => m.error) && (
              <button
                onClick={chatbot.retryLastMessage}
                className="p-1.5 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                title="Retry last message"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            )}
            {chatbot.messages.length > 0 && (
              <button
                onClick={chatbot.clearChat}
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
        {chatbot.messages.length === 0 ? (
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
            {chatbot.messages.map((message, i) => (
              <div
                key={i}
                className={cn(
                  "flex gap-3",
                  message.role === "user" ? "justify-end" : "justify-start"
                )}
              >
                {message.role !== "user" && (
                  <div
                    className={cn(
                      "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
                      message.error
                        ? "bg-red-100 dark:bg-red-900"
                        : "bg-blue-100 dark:bg-blue-900"
                    )}
                  >
                    <Bot
                      className={cn(
                        "w-5 h-5",
                        message.error
                          ? "text-red-600 dark:text-red-400"
                          : "text-blue-600 dark:text-blue-400"
                      )}
                    />
                  </div>
                )}
                <div
                  className={cn(
                    "max-w-[70%]",
                    message.role === "user" ? "order-1" : "order-2"
                  )}
                >
                  <Card
                    className={cn(
                      "py-0",
                      message.role === "user"
                        ? "bg-blue-600 text-white border-blue-600"
                        : message.error
                          ? "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800"
                          : "bg-gray-50 dark:bg-gray-700"
                    )}
                  >
                    <CardContent className="p-3">
                      <p
                        className={cn(
                          "whitespace-pre-wrap text-sm",
                          message.error && "text-red-700 dark:text-red-400"
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
                              : "text-gray-500 dark:text-gray-400"
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

            {chatbot.isLoading && (
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
            value={chatbot.input}
            onChange={(e) => chatbot.setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={chatbot.isLoading}
            className="flex-1"
            autoFocus
          />
          <Button
            type="submit"
            disabled={
              !chatbot.canSend || chatbot.isLoading || !chatbot.input.trim()
            }
            size="icon"
            title="Send message"
          >
            {!chatbot.canSend || chatbot.isLoading ? (
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
