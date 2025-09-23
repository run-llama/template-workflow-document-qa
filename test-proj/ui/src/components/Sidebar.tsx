import { X } from "lucide-react";
import { ScrollArea, cn } from "@llamaindex/ui";
import { useChatHistory, ChatHistory } from "../libs/chatHistory";

interface SidebarProps {
  className?: string;
}

export default function Sidebar({ className }: SidebarProps) {
  const { loading, getChats, selectedChatId, setSelectedChatId, deleteChat } =
    useChatHistory();
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
    setSelectedChatId(chat.handlerId);
  };

  const handleDeleteChat = (e: React.MouseEvent, handlerId: string): void => {
    e.stopPropagation();
    deleteChat(handlerId);
  };

  return (
    <div
      className={cn(
        "flex flex-col bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 sm:w-[300px]",
        className
      )}
    >
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <h3 className="text-sm font-medium text-gray-900 dark:text-white">
          Chats
        </h3>
      </div>

      {/* Chat List */}
      <ScrollArea className="flex-1">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Loading...
            </div>
          </div>
        ) : chats.length === 0 ? (
          <div className="flex items-center justify-center py-8">
            <div className="text-sm text-gray-500 dark:text-gray-400">
              No chats yet
            </div>
          </div>
        ) : (
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {chats.map((chat) => (
              <div
                key={chat.handlerId}
                className={cn(
                  "flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors",
                  selectedChatId === chat.handlerId
                    ? "bg-blue-50 dark:bg-blue-900/20"
                    : ""
                )}
                onClick={() => handleChatSelect(chat)}
              >
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-gray-900 dark:text-white truncate">
                    {formatTimestamp(chat.timestamp)}
                  </div>
                </div>
                <button
                  onClick={(e) => handleDeleteChat(e, chat.handlerId)}
                  className="ml-2 p-1 text-gray-400 hover:text-red-500 dark:text-gray-500 dark:hover:text-red-400 transition-colors"
                  aria-label="Delete chat"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
