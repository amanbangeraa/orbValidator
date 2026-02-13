import { Header } from "@/components/booking/header"
import { BookingWizard } from "@/components/booking/booking-wizard"
import { BookingProvider } from "@/lib/booking-context"

export default function PassportBookingPage() {
  return (
    <BookingProvider>
      <div className="min-h-screen bg-background">
        <Header />
        
        <main className="mx-auto max-w-5xl px-4 py-8 sm:px-6 lg:px-8">
          {/* Page title */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-foreground">
              Book Passport Appointment
            </h1>
            <p className="mt-2 text-lg text-muted-foreground">
              Schedule your visit to a Passport Seva Kendra in 5 easy steps
            </p>
          </div>

          {/* Booking wizard */}
          <BookingWizard />

          {/* Footer info */}
          <footer className="mt-12 border-t border-border pt-8">
            <div className="grid gap-6 sm:grid-cols-3 text-sm">
              <div>
                <h3 className="font-semibold text-foreground">Need Help?</h3>
                <p className="mt-1 text-muted-foreground">
                  Call our toll-free helpline at 1800-258-1800
                </p>
                <p className="text-muted-foreground">
                  Available 8 AM - 10 PM, all days
                </p>
              </div>
              <div>
                <h3 className="font-semibold text-foreground">Secure Process</h3>
                <p className="mt-1 text-muted-foreground">
                  Your data is protected with 256-bit SSL encryption and follows government security standards.
                </p>
              </div>
              <div>
                <h3 className="font-semibold text-foreground">Official Portal</h3>
                <p className="mt-1 text-muted-foreground">
                  This is the official Passport Seva Portal by the Ministry of External Affairs, Government of India.
                </p>
              </div>
            </div>
            <div className="mt-8 text-center text-xs text-muted-foreground">
              <p>Ministry of External Affairs | Government of India</p>
              <p className="mt-1">Passport Seva - Consular, Passport and Visa Division</p>
            </div>
          </footer>
        </main>
      </div>
    </BookingProvider>
  )
}
