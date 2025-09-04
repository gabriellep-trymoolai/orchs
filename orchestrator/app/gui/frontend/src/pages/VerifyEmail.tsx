// gui/frontend/src/pages/VerifyEmail.tsx
import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { AuthCard } from '@/components/AuthCard';
import { AuthInput } from '@/components/AuthInput';
import { Button } from '@/components/ui/button';
import { msal, apiScopes, msalReady } from '../auth/msal';

// Backend base (adjust if your API runs on a different port)
const BACKEND = (import.meta as any)?.env?.VITE_API_BASE_URL || 'http://localhost:8000';

export const VerifyEmail: React.FC = () => {
  const navigate = useNavigate();

  const [email, setEmail] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Handle the post-redirect flow and token acquisition
  useEffect(() => {
    (async () => {
      console.log('[VerifyEmail] effect start: waiting for MSAL ready…');
      setBusy(true);
      try {
        await msalReady;

        const accounts = msal.getAllAccounts();
        console.log('[VerifyEmail] accounts after redirect:', accounts);

        if (!accounts.length) {
          setError(null);
          return; // first visit; user will click the button
        }

        const result = await msal.acquireTokenSilent({
          account: accounts[0],
          scopes: apiScopes,
        });

        const token = result.accessToken;
        (window as any).__B2C_TOKEN__ = token;
        console.log('[VerifyEmail] acquired token (len):', token?.length);

        // Optional: backend sanity check
        const meResp = await fetch(`${BACKEND}/api/v1/me`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!meResp.ok) {
          const txt = await meResp.text();
          console.warn('[VerifyEmail] /api/v1/me non-200:', meResp.status, txt);
        }

        console.log('[VerifyEmail] navigating to /profile-setup');
        navigate('/profile-setup');
      } catch (e: any) {
        console.error('[VerifyEmail] effect error:', e);
        setError(e?.message || String(e));
      } finally {
        setBusy(false);
        console.log('[VerifyEmail] effect done');
      }
    })();
  }, [navigate]);

  // Click/submit → trigger B2C hosted UI
  const handleSubmit = async () => {
    console.log('[VerifyEmail] loginRedirect clicked. email:', email);
    setError(null);
    setBusy(true);
    try {
      await msalReady;
      await msal.loginRedirect({
        scopes: apiScopes,
        loginHint: email || undefined, // optional prefill on B2C UI
      });
      // Redirects away; no more code runs here
    } catch (e: any) {
      console.error('[VerifyEmail] loginRedirect error:', e);
      setBusy(false);
      setError(e?.message || String(e));
    }
  };

  return (
    <AuthCard>
      <div className="text-center mb-8">
        <h1 className="text-2xl font-semibold text-foreground mb-2">Verify your email</h1>
        <p className="text-sm text-muted-foreground">Enter the one-time code to create an account</p>
      </div>

      {error && <div className="mb-4 text-sm text-red-600">{error}</div>}

      {/* Submit handler on the form so Enter key works and clicks are guaranteed */}
      <form
        className="space-y-6"
        onSubmit={async (e) => {
          e.preventDefault();
          await handleSubmit();
        }}
      >
        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Email</label>
          <AuthInput
            type="email"
            value={email}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEmail(e.target.value)}
            placeholder="you@example.com"
            required
            disabled={busy}
            className="bg-input"
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Verification Code</label>
          <p className="mt-2 text-xs text-muted-foreground">Handled by B2C</p>
        </div>

        {/* type=submit so the form onSubmit fires */}
        <Button type="submit" className="w-full h-12" disabled={busy}>
          {busy ? 'Redirecting…' : 'Verify & Continue'}
        </Button>
      </form>

      <div className="text-center mt-6">
        <p className="text-sm text-muted-foreground">
          Already have an account?{' '}
          <Link to="/login" className="text-orange-primary hover:text-orange-light">
            Log in
          </Link>
        </p>
      </div>

      {/* Social button keeps the same redirect logic */}
      <Button
        variant="outline"
        className="w-full mt-4"
        type="button"
        onClick={async () => {
          console.log('[VerifyEmail] social login redirect');
          try {
            await msalReady;
            await msal.loginRedirect({ scopes: apiScopes });
          } catch (e: any) {
            console.error('[VerifyEmail] social login error:', e);
            setError(e?.message || String(e));
          }
        }}
        disabled={busy}
      >
        <div className="flex items-center gap-3">
          <div className="w-5 h-5 bg-white rounded-full flex items-center justify-center">
            <span className="text-xs font-bold text-blue-600">G</span>
          </div>
          Sign up with Google
        </div>
      </Button>
    </AuthCard>
  );
};
