import React from 'react';
import { MoolaiLogo } from './MoolaiLogo';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  MessageSquare, 
  Plus, 
  BarChart3, 
  Settings, 
  Users, 
  FileText, 
  Search,
  X,
  Key,
  Shield,
  Route,
  Building,
  Trash2,
  Server
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAppConversation, useAppSession } from '@/contexts/AppContext';

interface ChatSidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  isCollapsed?: boolean;
}

const navigationItems = [
  { icon: MessageSquare, label: 'New Chat', href: '/dashboard', isActive: false },
  { icon: Plus, label: 'New Task', href: '/dashboard', isActive: false },
  { icon: BarChart3, label: 'Analytics', href: '/analytics', isActive: false },
  { icon: Server, label: 'System Monitor', href: '/system-monitoring', isActive: false },
  { icon: Settings, label: 'Configuration', href: '/configuration', isActive: false, 
    submenu: [
      { icon: Key, label: 'API Keys', href: '/configuration/keys' },
      { icon: Route, label: 'LLM Router', href: '/configuration/router' },
      { icon: Shield, label: 'LLM Firewall', href: '/configuration/firewall' },
      
    ]
  },
  { icon: Users, label: 'Users', href: '/users', isActive: false },
  { icon: Building, label: 'Organizations', href: '/organizations', isActive: false },
  { icon: FileText, label: 'Logs', href: '/logs', isActive: false },
];

// Dynamic conversations will be loaded from the conversation service

export const ChatSidebar: React.FC<ChatSidebarProps> = ({ isOpen, onToggle, isCollapsed = false }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [expandedItems, setExpandedItems] = React.useState<string[]>(['Configuration']);
  
  // Conversation and session management
  const conversation = useAppConversation();
  const session = useAppSession();

  const handleNavigation = (href: string) => {
    navigate(href);
  };

  const toggleExpanded = (label: string) => {
    setExpandedItems(prev => 
      prev.includes(label) 
        ? prev.filter(item => item !== label)
        : [...prev, label]
    );
  };

  const handleNewChat = () => {
    if (session.userId) {
      const newConversation = conversation.createConversation(
        `Chat ${conversation.totalConversations + 1}`
      );
      if (newConversation) {
        navigate('/dashboard');
      }
    }
  };

  const handleConversationSelect = (conversationId: string) => {
    conversation.switchToConversation(conversationId);
    navigate('/dashboard');
  };

  const handleConversationDelete = (conversationId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    conversation.deleteConversation(conversationId);
  };

  const formatConversationName = (name: string, maxLength: number = 25) => {
    return name.length > maxLength ? `${name.slice(0, maxLength)}...` : name;
  };

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && !isCollapsed && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden" 
          onClick={onToggle}
        />
      )}
      
      {/* Sidebar */}
      <div className={cn(
        "fixed lg:relative inset-y-0 left-0 z-50 bg-sidebar border-r border-sidebar-border transform transition-all duration-200 ease-in-out",
        isCollapsed ? "w-16" : "w-72",
        !isCollapsed && (isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0")
      )}>
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-sidebar-border">
            <MoolaiLogo showText={!isCollapsed} />
            {!isCollapsed && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onToggle}
                className="lg:hidden text-sidebar-foreground hover:bg-sidebar-accent"
              >
                <X className="h-5 w-5" />
              </Button>
            )}
          </div>

          {/* Navigation */}
          <div className="flex-1 overflow-y-auto p-4 space-y-2">
            {navigationItems.map((item) => (
              <div key={item.label}>
                <Button
                  variant="ghost"
                  onClick={() => {
                    if (item.submenu) {
                      toggleExpanded(item.label);
                    } else if (item.label === 'New Chat') {
                      handleNewChat();
                    } else {
                      handleNavigation(item.href);
                    }
                  }}
                  className={cn(
                    "w-full justify-start h-10 text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                    location.pathname === item.href && "bg-sidebar-accent text-sidebar-accent-foreground",
                    isCollapsed && "justify-center px-2"
                  )}
                >
                  <item.icon className={cn("h-4 w-4", !isCollapsed && "mr-3")} />
                  {!isCollapsed && item.label}
                </Button>
                
                {/* Submenu */}
                {item.submenu && !isCollapsed && expandedItems.includes(item.label) && (
                  <div className="ml-4 mt-2 space-y-1">
                    {item.submenu.map((subItem) => (
                      <Button
                        key={subItem.label}
                        variant="ghost"
                        onClick={() => handleNavigation(subItem.href)}
                        className={cn(
                          "w-full justify-start h-8 text-sm text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                          location.pathname === subItem.href && "bg-sidebar-accent text-sidebar-accent-foreground"
                        )}
                      >
                        <subItem.icon className="h-3 w-3 mr-3" />
                        {subItem.label}
                      </Button>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {/* Conversations Section */}
            {!isCollapsed && (
              <div className="pt-6">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium text-sidebar-foreground">
                    Conversations ({conversation.totalConversations})
                  </h3>
                  <div className="flex items-center gap-1">
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className="text-sidebar-foreground hover:bg-sidebar-accent p-1"
                      onClick={handleNewChat}
                      disabled={!session.isConnected}
                    >
                      <Plus className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="sm" className="text-sidebar-foreground hover:bg-sidebar-accent p-1">
                      <Search className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                
                <div className="space-y-1 max-h-60 overflow-y-auto">
                  {conversation.conversationSummaries.length === 0 ? (
                    <div className="text-center py-4">
                      <p className="text-xs text-muted-foreground mb-2">No conversations yet</p>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={handleNewChat}
                        disabled={!session.isConnected}
                        className="text-xs"
                      >
                        Start New Chat
                      </Button>
                    </div>
                  ) : (
                    conversation.conversationSummaries.map((conv) => (
                      <div key={conv.id} className="group relative">
                        <Button
                          variant="ghost"
                          onClick={() => handleConversationSelect(conv.id)}
                          className={cn(
                            "w-full justify-start h-auto text-left text-sm text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground p-2",
                            conversation.currentConversation?.id === conv.id && "bg-sidebar-accent text-sidebar-accent-foreground"
                          )}
                        >
                          <div className="flex-1 min-w-0">
                            <div className="font-medium truncate">
                              {formatConversationName(conv.name)}
                            </div>
                            <div className="text-xs text-muted-foreground truncate">
                              {conv.last_message}
                            </div>
                            <div className="flex items-center justify-between mt-1">
                              <span className="text-xs text-muted-foreground">
                                {new Date(conv.updated_at).toLocaleDateString()}
                              </span>
                              <Badge variant="secondary" className="text-xs">
                                {conv.message_count}
                              </Badge>
                            </div>
                          </div>
                        </Button>
                        
                        {/* Delete button */}
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => handleConversationDelete(conv.id, e)}
                          className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity h-6 w-6 p-0 text-muted-foreground hover:text-destructive"
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Profile Section */}
          <div className="p-4 border-t border-sidebar-border">
            <Button 
              variant="ghost" 
              className={cn(
                "w-full justify-start h-10 text-sidebar-foreground hover:bg-sidebar-accent",
                isCollapsed && "justify-center px-2"
              )}
            >
              <div className={cn("w-8 h-8 bg-orange-primary rounded-full flex items-center justify-center", !isCollapsed && "mr-3")}>
                <span className="text-sm font-medium text-white">M</span>
              </div>
              {!isCollapsed && "Profile"}
            </Button>
          </div>
        </div>
      </div>
    </>
  );
};