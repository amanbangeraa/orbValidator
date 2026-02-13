"use client"

import { FileText, RefreshCw, Zap, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { HelpTooltip } from "@/components/booking/help-tooltip"
import { useBooking, type ServiceType } from "@/lib/booking-context"
import { cn } from "@/lib/utils"

const serviceOptions: {
  id: ServiceType
  title: string
  description: string
  icon: typeof FileText
  helpText: string
  price: string
  processingTime: string
}[] = [
  {
    id: "fresh",
    title: "Fresh Passport",
    description: "Apply for a new passport if you have never had one before",
    icon: FileText,
    helpText: "For first-time applicants who have never held an Indian passport",
    price: "Rs. 1,500",
    processingTime: "30-45 days",
  },
  {
    id: "reissue",
    title: "Passport Reissue",
    description: "Renew or replace your existing passport",
    icon: RefreshCw,
    helpText: "For renewal, name change, address change, or replacing damaged/lost passports",
    price: "Rs. 1,500",
    processingTime: "15-30 days",
  },
  {
    id: "tatkaal-fresh",
    title: "Tatkaal Fresh Passport",
    description: "Urgent processing for first-time applicants",
    icon: Zap,
    helpText: "Tatkaal service provides faster processing for urgent requirements. Additional fee applies.",
    price: "Rs. 3,500",
    processingTime: "1-3 days",
  },
  {
    id: "tatkaal-reissue",
    title: "Tatkaal Reissue",
    description: "Urgent processing for passport renewal or replacement",
    icon: Zap,
    helpText: "Tatkaal service for existing passport holders who need urgent renewal",
    price: "Rs. 3,500",
    processingTime: "1-3 days",
  },
]

export function ServiceSelection() {
  const { bookingData, updateServiceType, setCurrentStep, canProceed } = useBooking()

  const handleContinue = () => {
    if (canProceed(1)) {
      setCurrentStep(2)
    }
  }

  return (
    <div className="space-y-8">
      {/* Page header */}
      <div>
        <h2 className="text-2xl font-semibold text-foreground text-balance">
          What type of passport service do you need?
        </h2>
        <p className="mt-2 text-muted-foreground">
          Select the service that best matches your requirements. Not sure?{" "}
          <button className="text-primary underline underline-offset-2 hover:text-primary/80">
            Help me choose
          </button>
        </p>
      </div>

      {/* Service options */}
      <RadioGroup
        value={bookingData.serviceType || ""}
        onValueChange={(value) => updateServiceType(value as ServiceType)}
        className="grid gap-4 md:grid-cols-2"
      >
        {serviceOptions.map((service) => {
          const Icon = service.icon
          const isSelected = bookingData.serviceType === service.id
          const isTatkaal = service.id?.includes("tatkaal")

          return (
            <Label
              key={service.id}
              htmlFor={service.id || ""}
              className="cursor-pointer"
            >
              <Card
                className={cn(
                  "relative transition-all hover:border-primary/50",
                  isSelected && "border-primary ring-2 ring-primary/20",
                  isTatkaal && "border-accent/30"
                )}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div
                        className={cn(
                          "flex h-10 w-10 items-center justify-center rounded-lg",
                          isTatkaal ? "bg-accent/10 text-accent" : "bg-primary/10 text-primary"
                        )}
                      >
                        <Icon className="h-5 w-5" aria-hidden="true" />
                      </div>
                      <div>
                        <CardTitle className="text-base font-medium flex items-center gap-2">
                          {service.title}
                          <HelpTooltip content={service.helpText} />
                        </CardTitle>
                        {isTatkaal && (
                          <span className="inline-flex items-center rounded-full bg-accent/10 px-2 py-0.5 text-xs font-medium text-accent-foreground">
                            Urgent
                          </span>
                        )}
                      </div>
                    </div>
                    <RadioGroupItem
                      value={service.id || ""}
                      id={service.id || ""}
                      className="mt-1"
                    />
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  <CardDescription className="text-sm">
                    {service.description}
                  </CardDescription>
                  <div className="mt-4 flex items-center gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Fee: </span>
                      <span className="font-medium text-foreground">{service.price}</span>
                    </div>
                    <div className="h-4 w-px bg-border" aria-hidden="true" />
                    <div>
                      <span className="text-muted-foreground">Processing: </span>
                      <span className="font-medium text-foreground">{service.processingTime}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </Label>
          )
        })}
      </RadioGroup>

      {/* What happens next */}
      <Card className="bg-muted/50 border-muted">
        <CardContent className="flex items-start gap-4 py-4">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
            <span className="text-sm font-medium">?</span>
          </div>
          <div>
            <h3 className="font-medium text-foreground">What happens next?</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              After selecting your service type, you will provide your personal details, choose an
              appointment slot at a Passport Seva Kendra near you, and complete the payment online.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Continue button */}
      <div className="flex justify-end pt-4">
        <Button
          size="lg"
          onClick={handleContinue}
          disabled={!canProceed(1)}
          className="gap-2 px-8"
        >
          Continue to Details
          <ArrowRight className="h-4 w-4" aria-hidden="true" />
        </Button>
      </div>
    </div>
  )
}
