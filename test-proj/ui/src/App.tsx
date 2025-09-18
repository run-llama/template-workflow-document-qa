import { ApiProvider } from "@llamaindex/ui";
import Home from "./pages/Home";
import { Theme } from "@radix-ui/themes";
import { clients } from "@/libs/clients";

export default function App() {
  return (
    <Theme>
      <ApiProvider clients={clients}>
        <Home />
      </ApiProvider>
    </Theme>
  );
}
