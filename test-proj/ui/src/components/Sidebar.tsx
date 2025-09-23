import { MessageSquare, Clock, Loader2 } from "lucide-react";
import {
  ScrollArea,
  Card,
  CardContent,
  cn,
} from "@llamaindex/ui";
import { useChatHistory, ChatHistory } from "../libs/chatHistory";

interface SidebarProps {
  className?: string;
}

export default function Sidebar({ className }: SidebarProps) {
  const { loading, getChats, selectedChat, setSelectedChat } = useChatHistory();
  const chats = getChats();

  const formatTimestamp = (timestamp: string): string => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60);

    if (diffInHours < 1) {
      return "Just now";
    } else if (diffInHours < 24) {
      return `${Math.floor(diffInHours)}h ago`;
    } else if (diffInHours < 24 * 7) {
      return `${Math.floor(diffInHours / 24)}d ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  const handleChatSelect = (chat: ChatHistory): void => {
    setSelectedChat(chat);
  };

  return (
    <div className={cn("flex flex-col h-full bg-white dark:bg-gray-800", className)}>
      {/* Header */}
      <div className="px-4 py-3 border-b dark:border-gray-700">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          <h3 className="font-semibold text-gray-900 dark:text-white">
            Chat History
          </h3>
          {loading && (
            <Loader2 className="w-4 h-4 animate-spin text-gray-500 dark:text-gray-400" />
          )}
        </div>
      </div>

      {/* Chat List */}
      <ScrollArea className="flex-1">
        {loading ? (
          <div className="flex items-center justify-center h-full min-h-[200px]">
            <div className="text-center">
              <Loader2 className="w-8 h-8 text-gray-400 dark:text-gray-600 mx-auto mb-3 animate-spin" />
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Loading chat history...
              </p>
            </div>
          </div>
        ) : chats.length === 0 ? (
          <div className="flex items-center justify-center h-full min-h-[200px]">
            <div className="text-center">
              <MessageSquare className="w-8 h-8 text-gray-400 dark:text-gray-600 mx-auto mb-3" />
              <p className="text-sm text-gray-500 dark:text-gray-400">
                No chat history yet
              </p>
              <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                Start a conversation to see it here
              </p>
            </div>
          </div>
        ) : (
          <div className="p-2 space-y-2">
            {chats.map((chat) => (
              <Card
                key={chat.handlerId}
                className={cn(
                  "cursor-pointer transition-all duration-200 hover:shadow-md",
                  selectedChat?.handlerId === chat.handlerId
                    ? "bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 shadow-sm"
                    : "hover:bg-gray-50 dark:hover:bg-gray-700"
                )}
                onClick={() => handleChatSelect(chat)}
              >
                <CardContent className="p-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <MessageSquare className={cn(
                          "w-4 h-4 flex-shrink-0",
                          selectedChat?.handlerId === chat.handlerId
                            ? "text-blue-600 dark:text-blue-400"
                            : "text-gray-500 dark:text-gray-400"
                        )} />
                        <p className={cn(
                          "text-sm font-medium truncate",
                          selectedChat?.handlerId === chat.handlerId
                            ? "text-blue-900 dark:text-blue-100"
                            : "text-gray-900 dark:text-white"
                        )}>
                          Chat {chat.handlerId.slice(-8)}
                        </p>
                      </div>
                      <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
                        <Clock className="w-3 h-3" />
                        <span>{formatTimestamp(chat.timestamp)}</span>
                      </div>
                    </div>
                    {selectedChat?.handlerId === chat.handlerId && (
                      <div className="w-2 h-2 bg-blue-600 dark:bg-blue-400 rounded-full flex-shrink-0 mt-1" />
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
