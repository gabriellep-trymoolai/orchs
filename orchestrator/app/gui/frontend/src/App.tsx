import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppProvider } from "./contexts/AppContext";
import { SignUp } from "./pages/SignUp";
import { Login } from "./pages/Login";
import { VerifyEmail } from "./pages/VerifyEmail";
import { ProfileSetup } from "./pages/ProfileSetup";
import { LLMSetup } from "./pages/LLMSetup";
import { Dashboard } from "./pages/Dashboard";
import { AnalyticsDashboard } from "./pages/AnalyticsDashboard";
import { SystemMonitoringDashboard } from "./pages/SystemMonitoringDashboard";
import { ConfigurationKeys } from "./pages/ConfigurationKeys";
import { ConfigurationFirewall } from "./pages/ConfigurationFirewall";
import { Logs } from "./pages/Logs";
import { Users } from "./pages/Users";
import { Organizations } from "./pages/Organizations";
import { WebSocketTest } from "./pages/WebSocketTest";
import { MainLayout } from "./components/MainLayout";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <AppProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<SignUp />} />
            <Route path="/signup" element={<SignUp />} />
            <Route path="/login" element={<Login />} />
            <Route path="/verify-email" element={<VerifyEmail />} />
            <Route path="/profile-setup" element={<ProfileSetup />} />
            <Route path="/llm-setup" element={<LLMSetup />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/analytics" element={<MainLayout><AnalyticsDashboard /></MainLayout>} />
            <Route path="/system-monitoring" element={<MainLayout><SystemMonitoringDashboard /></MainLayout>} />
            <Route path="/configuration/keys" element={<MainLayout><ConfigurationKeys /></MainLayout>} />
            <Route path="/configuration/router" element={<MainLayout><div className="p-6"><h1 className="text-2xl font-semibold">LLM Router Configuration</h1><p className="text-muted-foreground mt-2">Router configuration will be implemented here.</p></div></MainLayout>} />
            <Route path="/configuration/firewall" element={<MainLayout><ConfigurationFirewall /></MainLayout>} />
            
            <Route path="/logs" element={<MainLayout><Logs /></MainLayout>} />
            <Route path="/users" element={<MainLayout><Users /></MainLayout>} />
            <Route path="/organizations" element={<MainLayout><Organizations /></MainLayout>} />
            <Route path="/test-websocket" element={<WebSocketTest />} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </AppProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
