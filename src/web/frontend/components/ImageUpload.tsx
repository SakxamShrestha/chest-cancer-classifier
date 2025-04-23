// src/web/frontend/components/ImageUpload.tsx
import React, { useState, useRef } from 'react';
import { 
  Box, 
  Button, 
  Text, 
  VStack, 
  Image, 
  useToast,
  Icon,
  Flex,
} from '@chakra-ui/react';
import { keyframes } from '@emotion/react';
import { FiUploadCloud, FiX, FiImage } from 'react-icons/fi';

const pulseKeyframe = keyframes`
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
`;

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  isLoading: boolean;
}

export const ImageUpload: React.FC<ImageUploadProps> = ({ onImageSelect, isLoading }) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const toast = useToast();

  const pulseAnimation = `${pulseKeyframe} 1.5s ease-in-out infinite`;

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

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

    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
    onImageSelect(file);
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(false);
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

  return (
    <VStack spacing={6} width="100%" maxW="800px" mx="auto">
      <Box
        width="100%"
        height="400px"
        border="3px dashed"
        borderColor={isDragging ? 'blue.400' : 'blue.200'}
        borderRadius="xl"
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        onDrop={handleDrop}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        position="relative"
        overflow="hidden"
        bg="gray.50"
        transition="all 0.3s ease"
        _hover={{ borderColor: 'blue.400' }}
        sx={{
          animation: isDragging ? pulseAnimation : 'none'
        }}
      >
        {preview ? (
          <Flex position="relative" width="100%" height="100%">
            <Image
              src={preview}
              alt="Preview"
              maxHeight="100%"
              objectFit="contain"
              p={4}
            />
            <Button
              position="absolute"
              top={2}
              right={2}
              colorScheme="red"
              size="sm"
              onClick={() => {
                setPreview(null);
                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                }
              }}
              leftIcon={<Icon as={FiX} />}
            >
              Remove
            </Button>
          </Flex>
        ) : (
          <VStack spacing={4} p={8}>
            <Icon
              as={FiUploadCloud}
              w={12}
              h={12}
              color="blue.400"
            />
            <Text
              color="gray.600"
              fontSize="lg"
              textAlign="center"
              fontWeight="medium"
            >
              Drag and drop your CT scan image here
            </Text>
            <Text color="gray.500" fontSize="sm">
              or
            </Text>
            <Button
              colorScheme="blue"
              size="lg"
              onClick={() => fileInputRef.current?.click()}
              isLoading={isLoading}
              leftIcon={<Icon as={FiImage} />}
              shadow="md"
              _hover={{
                transform: 'translateY(-2px)',
                shadow: 'lg',
              }}
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
    </VStack>
  );
};