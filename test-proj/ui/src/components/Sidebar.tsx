import { Plus, X } from "lucide-react";
import { Button, ScrollArea, cn } from "@llamaindex/ui";
import { ChatHistory, UseChatHistory } from "../libs/useChatHistory";

interface SidebarProps {
  className?: string;
  chatHistory: UseChatHistory;
}

export default function Sidebar({ className, chatHistory }: SidebarProps) {
  const {
    loading,
    chats,
    selectedChatId,
    setSelectedChatId,
    deleteChat,
    createNewChat,
  } = chatHistory;

  const formatTimestamp = (timestamp: string): string => {
    const date = new Date(timestamp);
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();

    const timeString = date.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    if (isToday) {
      return timeString;
    } else {
      const dateString = date.toLocaleDateString();
      return `${dateString} ${timeString}`;
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
        "flex flex-col bg-sidebar border-r border-sidebar-border w-[280px]",
        className,
      )}
    >
      {/* Header */}
      <div className="px-4 py-4 border-b border-sidebar-border">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-sidebar-foreground">
            Chats
          </h3>
          <Button
            size="sm"
            variant="ghost"
            onClick={createNewChat}
            className="h-8 w-8 p-0 hover:bg-sidebar-accent"
            title="New Chat"
          >
            <Plus className="w-4 h-4 text-sidebar-foreground" />
          </Button>
        </div>
      </div>

      {/* Chat List */}
      <ScrollArea className="flex-1">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="text-sm text-muted-foreground">
              Loading...
            </div>
          </div>
        ) : chats.length === 0 ? (
          <div className="flex items-center justify-center py-8">
            <div className="text-sm text-muted-foreground">
              No chats yet
            </div>
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {chats.map((chat) => (
              <div
                key={chat.handlerId}
                className={cn(
                  "flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer hover:bg-sidebar-accent transition-colors",
                  selectedChatId === chat.handlerId
                    ? "bg-sidebar-primary text-sidebar-primary-foreground"
                    : "text-sidebar-foreground",
                )}
                onClick={() => handleChatSelect(chat)}
              >
                <div className="flex-1 min-w-0">
                  <div className="text-sm truncate">
                    {formatTimestamp(chat.timestamp)}
                  </div>
                </div>
                <button
                  onClick={(e) => handleDeleteChat(e, chat.handlerId)}
                  className={cn(
                    "ml-2 p-1 rounded transition-colors",
                    selectedChatId === chat.handlerId
                      ? "text-sidebar-primary-foreground/70 hover:text-sidebar-primary-foreground hover:bg-sidebar-primary-foreground/10"
                      : "text-muted-foreground hover:text-destructive"
                  )}
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
