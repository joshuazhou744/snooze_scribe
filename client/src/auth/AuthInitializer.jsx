import { useCallback, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import { useAuth0 } from '@auth0/auth0-react';
import { setAuthHandlers } from '../services/apiClient';

const AUTH0_AUDIENCE = import.meta.env.VITE_AUTH0_AUDIENCE;

const isAuthError = (error) => {
  const status = error?.response?.status;
  if (status === 401 || status === 403) {
    return true;
  }
  const errorCode = error?.error || error?.code;
  return (
    errorCode === 'login_required' ||
    errorCode === 'consent_required' ||
    errorCode === 'invalid_grant' ||
    errorCode === 'missing_refresh_token' ||
    errorCode === 'expired_token'
  );
};

export default function AuthInitializer({ children = null }) {
  const { getAccessTokenSilently, logout, isAuthenticated } = useAuth0();
  const authFailureRef = useRef(false);

  const handleAuthFailure = useCallback(
    (error) => {
      if (!isAuthenticated || authFailureRef.current) {
        return;
      }
      if (!isAuthError(error)) {
        return;
      }
      authFailureRef.current = true;
      logout({
        logoutParams: {
          returnTo: window.location.origin,
        },
      });
    },
    [isAuthenticated, logout]
  );

  useEffect(() => {
    setAuthHandlers({
      getToken: () => getAccessTokenSilently({ audience: AUTH0_AUDIENCE }),
      onAuthFailure: handleAuthFailure,
    });
    return () => {
      setAuthHandlers({ getToken: null, onAuthFailure: null });
    };
  }, [getAccessTokenSilently, handleAuthFailure]);

  useEffect(() => {
    if (isAuthenticated) {
      authFailureRef.current = false;
    }
  }, [isAuthenticated]);

  return children;
}

AuthInitializer.propTypes = {
  children: PropTypes.node,
};
