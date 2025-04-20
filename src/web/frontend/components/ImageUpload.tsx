// src/web/frontend/components/ImageUpload.tsx
import React, { useState, useRef } from 'react';
import { Box, Button, Text, VStack, Image, useToast } from '@chakra-ui/react';

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  isLoading: boolean;
}

export const ImageUpload: React.FC<ImageUploadProps> = ({ onImageSelect, isLoading }) => {
  const [preview, setPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const toast = useToast();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      toast({
        title: 'Invalid file type',
        description: 'Please upload an image file',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);

    onImageSelect(file);
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
      onImageSelect(file);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  return (
    <VStack spacing={4} width="100%">
      <Box
        width="100%"
        height="300px"
        border="2px dashed"
        borderColor="gray.300"
        borderRadius="md"
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        position="relative"
        overflow="hidden"
        bg="gray.50"
      >
        {preview ? (
          <Image
            src={preview}
            alt="Preview"
            maxHeight="100%"
            objectFit="contain"
          />
        ) : (
          <VStack spacing={2}>
            <Text color="gray.500">
              Drag and drop your CT scan image here or click to select
            </Text>
            <Button
              colorScheme="blue"
              onClick={() => fileInputRef.current?.click()}
              isLoading={isLoading}
            >
              Select Image
            </Button>
          </VStack>
        )}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          accept="image/*"
          style={{ display: 'none' }}
        />
      </Box>
      {preview && (
        <Button
          colorScheme="red"
          variant="outline"
          onClick={() => {
            setPreview(null);
            if (fileInputRef.current) {
              fileInputRef.current.value = '';
            }
          }}
        >
          Clear Image
        </Button>
      )}
    </VStack>
  );
};