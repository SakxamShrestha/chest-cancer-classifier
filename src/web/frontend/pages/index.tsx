// src/web/frontend/pages/index.tsx
import React, { useState } from 'react';
import { Box, Grid, useToast } from '@chakra-ui/react';
import { ImageUpload } from '../components/ImageUpload';
import { Prediction } from '../components/Prediction';
import type { PredictionResponse } from '../types/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();

  const handleImageSelect = async (file: File) => {
    setIsLoading(true);
    setPrediction(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData,
        headers: {
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.detail || 
          `Server error: ${response.status} ${response.statusText}`
        );
      }

      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error('Error details:', error);
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to analyze image',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Grid
      templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)' }}
      gap={8}
      width="100%"
    >
      <Box>
        <ImageUpload onImageSelect={handleImageSelect} isLoading={isLoading} />
      </Box>
      <Box>
        <Prediction prediction={prediction} isLoading={isLoading} />
      </Box>
    </Grid>
  );
}