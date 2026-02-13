import { Shield, HelpCircle, Phone } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

export function Header() {
  return (
    <header className="border-b border-border bg-card">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo and Title */}
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
              <Shield className="h-6 w-6 text-primary-foreground" aria-hidden="true" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-foreground">Passport Seva</h1>
              <p className="text-xs text-muted-foreground">Ministry of External Affairs</p>
            </div>
          </div>

          {/* Help and Contact */}
          <div className="flex items-center gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="sm" className="gap-2 text-muted-foreground hover:text-foreground">
                    <Phone className="h-4 w-4" aria-hidden="true" />
                    <span className="hidden sm:inline">1800-258-1800</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Toll-free helpline (8 AM - 10 PM)</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="sm" className="gap-2 text-muted-foreground hover:text-foreground">
                    <HelpCircle className="h-4 w-4" aria-hidden="true" />
                    <span className="hidden sm:inline">Help</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Get help with your application</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>
      </div>
    </header>
  )
}
